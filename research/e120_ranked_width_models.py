#!/usr/bin/env python3
"""Re-weight the measured Route B gain onto the ranked prompts' width regime.

    usage: research/e120_ranked_width_models.py [--gate-price PATH] [--out PATH]

The local fixture is a copy task. It drafts far deeper than the eight hidden
ranked prompts, and the per-width gain table is strongly non-monotonic, so the
locally measured leg effect is not transferable without re-weighting.

Three models, all `harness=ranked` estimates built from `harness=local`
measurements, per E120 feedback 9:

  A  measured local     the session's own realised width histogram
  B  ranked point       each ranked prompt's mean verify width, the gain table
                        interpolated linearly between integer widths
  C  adverse bracket    each ranked prompt's mass moved to whichever of
                        floor(M) and ceil(M) gives the LOWER gain

Every value derived from a mean rather than a measured histogram is inferred,
and the output marks it. The ranked width and weight table is advisor-supplied
from the live board; this module does not measure it and cannot check it.

WHY THE LEG SHARE IS INTERPOLATED RATHER THAN FIXED. Campaign finding F95 says
the wide-QMV share of the leg rises with verify width, so `WIDE_QMV_TO_LEG` is
a coefficient with a width label and not a constant. Two points are known:
E116 fitted 0.6070 across twelve legs, and rung 5e inverts to 0.796 at mean
width 7.359. Two points void the constant and do not replace it, so this
interpolates between them and labels the result inferred rather than measured.
"""

from __future__ import annotations

import argparse
import json
import pathlib

SHIPPED_GATE = "m4_and_volume_100k"
LEG_TO_RANKED = 0.95

# Advisor-supplied, from the live board. `weight` is the prompt's marginal
# weight on the published median over eight prompts.
RANKED_PROMPTS = {
    "beagle": {"mean_width": 5.382, "accept_rate": 0.834, "weight": 0.4862},
    "medicine": {"mean_width": 6.256, "accept_rate": 0.892, "weight": 0.2508},
    "essays": {"mean_width": 6.087, "accept_rate": 0.897, "weight": 0.1598},
    "botany": {"mean_width": 7.148, "accept_rate": 0.865, "weight": 0.0124},
    "republic": {"mean_width": 5.989, "accept_rate": 0.903, "weight": 0.0100},
    "travel": {"mean_width": 3.656, "accept_rate": 0.533, "weight": 0.0},
    "drama": {"mean_width": 3.298, "accept_rate": 0.449, "weight": 0.0},
    "plutarch": {"mean_width": 1.154, "accept_rate": 0.333, "weight": 0.0},
}

# Rung 5e/5g realised histogram, from the timed legs.
SESSION_HISTOGRAM = {2: 4, 4: 16, 5: 20, 6: 20, 7: 12, 8: 240}

# F95 anchors: (mean verify width, wide-QMV share of the leg).
SHARE_ANCHOR_LOW = (5.382, 0.607)   # E116, twelve legs
SHARE_ANCHOR_HIGH = (7.359, 0.796)  # rung 5e, inverted from the measurement

# Inputs-per-group for each verify width, from the dispatch switch. E121 shares
# the chunk-sum tree only when IPG <= 4, and it deletes `H = IPG / 2` of the
# tree in integer division: a third at IPG 3, a half at IPG 4, nothing at 5.
WIDTH_TO_IPG = {3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3}
# Askeladd's isolated capture: `x_sumshoist` reached 90.2 % of the ceiling that
# deletes the term outright, so the full sum-work share is gain / 0.902.
SUMSHOIST_CAPTURE = 0.902
# Rung 5g measured E121's in-situ leg effect on the control arm directly.
E121_INSITU_LEG_PCT = 0.4065


def e121_deleted_fraction(width: int) -> float:
    ipg = WIDTH_TO_IPG.get(width, 0)
    if ipg > 4:
        return 0.0
    return (ipg // 2) / ipg if ipg else 0.0


def leg_share(width: float) -> float:
    """Interpolate the wide-QMV leg share between the two F95 anchors."""
    (w0, s0), (w1, s1) = SHARE_ANCHOR_LOW, SHARE_ANCHOR_HIGH
    return s0 + (width - w0) * (s1 - s0) / (w1 - w0)


def load_gate_price(path: pathlib.Path) -> dict[int, dict]:
    raw = json.loads(path.read_text())
    table = {}
    for key, row in raw.items():
        gate = row["gates"][SHIPPED_GATE]
        table[int(key)] = {
            "base_us": row["base_us"],
            "net_us": gate["net_us"],
            "pct": gate["pct"],
        }
    return table


def histogram_gain(table: dict[int, dict], histogram: dict[int, float]) -> float:
    """Time-weighted wide-QMV gain over a width histogram.

    Rounds at different widths cost different amounts, so the correct weight is
    round count times that width's wide-QMV time, not round count alone.
    """
    base = saved = 0.0
    for width, count in histogram.items():
        if width not in table:
            continue
        base += count * table[width]["base_us"]
        saved += count * table[width]["net_us"]
    return 100.0 * saved / base if base else 0.0


def interpolated_gain(table: dict[int, dict], width: float) -> float:
    lo, hi = int(width), int(width) + 1
    if lo not in table:
        return 0.0
    if hi not in table:
        return table[lo]["pct"]
    frac = width - lo
    return table[lo]["pct"] * (1 - frac) + table[hi]["pct"] * frac


def adverse_gain(table: dict[int, dict], width: float) -> tuple[float, int]:
    lo, hi = int(width), int(width) + 1
    options = [w for w in (lo, hi) if w in table]
    if not options:
        return 0.0, lo
    pick = min(options, key=lambda w: table[w]["pct"])
    return table[pick]["pct"], pick


def post_merge_table(table: dict[int, dict],
                     histogram: dict[int, float]) -> tuple[dict[int, dict], dict]:
    """Reprice each width against the E121 control instead of stock MLX.

    The rung-5d grid measured Route B against the pre-E121 incumbent, so it
    overstates the gain at every width where E121 already shares the tree. The
    shape of E121's per-width gain is modelled as `deleted_fraction(M) *
    full_sum_share(M)`, and a SINGLE free scale is fitted so that the
    histogram-weighted total reproduces the leg effect rung 5g measured on the
    control arm. The total is therefore fitted, not predicted; only the shape
    across widths is a prediction, and it is inferred, not measured.
    """
    shape = {w: e121_deleted_fraction(w) * row["pct"] / SUMSHOIST_CAPTURE
             for w, row in table.items()}
    unscaled = histogram_gain(
        {w: {"base_us": table[w]["base_us"],
             "net_us": table[w]["base_us"] * shape[w] / 100.0}
         for w in table}, histogram)
    target_qmv = E121_INSITU_LEG_PCT / SHARE_ANCHOR_HIGH[1]
    scale = target_qmv / unscaled if unscaled else 0.0

    priced = {}
    for width, row in table.items():
        e121 = scale * shape[width]
        residual = (row["pct"] - e121) / (1.0 - e121 / 100.0)
        priced[width] = {
            "base_us": row["base_us"],
            "pct": residual,
            "net_us": row["base_us"] * residual / 100.0,
            "e121_gain_pct": e121,
        }
    diagnostics = {
        "e121_insitu_leg_pct_measured": E121_INSITU_LEG_PCT,
        "e121_insitu_wide_qmv_pct_implied": target_qmv,
        "shape_scale_fitted": scale,
        "note": ("one free scale fitted to one measured number; the width "
                 "shape is inferred from the E121 source, not measured"),
    }
    return priced, diagnostics


def ranked_model(table: dict[int, dict], adverse: bool) -> dict:
    """Weighted ranked estimate over the drafting prompts."""
    total_weight = sum(p["weight"] for p in RANKED_PROMPTS.values())
    rows = {}
    leg_sum = 0.0
    qmv_sum = 0.0
    for name, prompt in RANKED_PROMPTS.items():
        if prompt["weight"] <= 0.0:
            continue
        width = prompt["mean_width"]
        if adverse:
            gain, at_width = adverse_gain(table, width)
        else:
            gain, at_width = interpolated_gain(table, width), None
        # The share is always read at the prompt's own mean width. Model C
        # moves only the gain, so the two models differ in one thing.
        share = leg_share(width)
        weight = prompt["weight"] / total_weight
        rows[name] = {
            "mean_width": width,
            "weight_renormalised": weight,
            "wide_qmv_gain_pct": gain,
            "gain_read_at_integer_width": at_width,
            "leg_share_interpolated": share,
            "leg_gain_pct": gain * share,
        }
        qmv_sum += weight * gain
        leg_sum += weight * gain * share
    return {
        "prompts": rows,
        "weighted_wide_qmv_gain_pct": qmv_sum,
        "weighted_leg_gain_pct": leg_sum,
        "weighted_ranked_gain_pct": leg_sum * LEG_TO_RANKED,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-price", type=pathlib.Path,
                        default=pathlib.Path("research/out/e120-gate-price.json"))
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("research/out/e120-ranked-width-models.json"))
    args = parser.parse_args()

    table = load_gate_price(args.gate_price)
    local_qmv = histogram_gain(table, SESSION_HISTOGRAM)
    local_share = leg_share(SHARE_ANCHOR_HIGH[0])

    model_b = ranked_model(table, adverse=False)
    model_c = ranked_model(table, adverse=True)
    ratio = (model_b["weighted_leg_gain_pct"] / model_c["weighted_leg_gain_pct"]
             if model_c["weighted_leg_gain_pct"] else float("inf"))

    merged, merged_diag = post_merge_table(table, SESSION_HISTOGRAM)
    model_b_pm = ranked_model(merged, adverse=False)
    model_c_pm = ranked_model(merged, adverse=True)
    ratio_pm = (model_b_pm["weighted_leg_gain_pct"]
                / model_c_pm["weighted_leg_gain_pct"]
                if model_c_pm["weighted_leg_gain_pct"] else float("inf"))
    # Cruder alternative treatment: scale the pre-merge models by the single
    # session-level attenuation rung 5g measured. It is conservative at ranked
    # widths, because the session sits at M=8 where E121 shares and the ranked
    # prompts sit near M=5 where it does not.
    uniform = histogram_gain(merged, SESSION_HISTOGRAM) / local_qmv

    result = {
        "post_merge_control": {
            "per_width_table": {str(k): v for k, v in sorted(merged.items())},
            "diagnostics": merged_diag,
            "model_a_wide_qmv_gain_pct": histogram_gain(merged, SESSION_HISTOGRAM),
            "model_a_predicted_leg_gain_pct":
                histogram_gain(merged, SESSION_HISTOGRAM) * local_share,
            "model_b_ranked_point": model_b_pm,
            "model_c_adverse_bracket": model_c_pm,
            "b_over_c_leg_ratio": ratio_pm,
            "b_over_c_exceeds_1_5": ratio_pm > 1.5,
            "uniform_attenuation_alternative": {
                "factor": uniform,
                "model_b_ranked_gain_pct":
                    model_b["weighted_ranked_gain_pct"] * uniform,
                "model_c_ranked_gain_pct":
                    model_c["weighted_ranked_gain_pct"] * uniform,
            },
        },
        "harness": "ranked",
        "basis": "harness=local per-width gate price, re-weighted",
        "inferred": ("models B and C use advisor-supplied ranked mean widths, "
                     "not measured per-prompt histograms; the leg share is "
                     "interpolated between two F95 anchors"),
        "leg_to_ranked": LEG_TO_RANKED,
        "share_anchors": {"low": SHARE_ANCHOR_LOW, "high": SHARE_ANCHOR_HIGH},
        "per_width_table": {str(k): v for k, v in sorted(table.items())},
        "model_a_measured_local": {
            "histogram": {str(k): v for k, v in sorted(SESSION_HISTOGRAM.items())},
            "wide_qmv_gain_pct": local_qmv,
            "leg_share_at_mean_width": local_share,
            "predicted_leg_gain_pct": local_qmv * local_share,
        },
        "model_b_ranked_point": model_b,
        "model_c_adverse_bracket": model_c,
        "b_over_c_leg_ratio": ratio,
        "b_over_c_exceeds_1_5": ratio > 1.5,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("per-width wide-QMV gain %%: %s"
          % {k: round(v["pct"], 3) for k, v in sorted(table.items())})
    print("A measured local   wide-QMV %.3f %%  share %.3f  leg %.3f %%"
          % (local_qmv, local_share, local_qmv * local_share))
    for tag, model in (("B ranked point   ", model_b),
                       ("C adverse bracket", model_c)):
        print("%s  wide-QMV %.3f %%  leg %.3f %%  ranked %.3f %%"
              % (tag, model["weighted_wide_qmv_gain_pct"],
                 model["weighted_leg_gain_pct"],
                 model["weighted_ranked_gain_pct"]))
    print("B/C leg ratio %.3f  (loud if > 1.5: %s)"
          % (ratio, result["b_over_c_exceeds_1_5"]))

    print()
    print("=== repriced against the E121 control (post-merge) ===")
    print("E121 in-situ leg %.4f %% -> wide-QMV %.4f %%, shape scale %.4f"
          % (merged_diag["e121_insitu_leg_pct_measured"],
             merged_diag["e121_insitu_wide_qmv_pct_implied"],
             merged_diag["shape_scale_fitted"]))
    print("per-width post-merge gain %%: %s"
          % {k: round(v["pct"], 3) for k, v in sorted(merged.items())})
    pm_a = histogram_gain(merged, SESSION_HISTOGRAM)
    print("A measured local   wide-QMV %.3f %%  leg %.3f %% predicted"
          % (pm_a, pm_a * local_share))
    for tag, model in (("B ranked point   ", model_b_pm),
                       ("C adverse bracket", model_c_pm)):
        print("%s  wide-QMV %.3f %%  leg %.3f %%  ranked %.3f %%"
              % (tag, model["weighted_wide_qmv_gain_pct"],
                 model["weighted_leg_gain_pct"],
                 model["weighted_ranked_gain_pct"]))
    print("B/C leg ratio %.3f  (loud if > 1.5: %s)"
          % (ratio_pm, ratio_pm > 1.5))
    print("uniform-attenuation alternative x%.4f: B ranked %.3f %%  C ranked %.3f %%"
          % (uniform, model_b["weighted_ranked_gain_pct"] * uniform,
             model_c["weighted_ranked_gain_pct"] * uniform))
    for name, row in model_b["prompts"].items():
        c_row = model_c["prompts"][name]
        print("  %-9s M=%.3f w=%.4f  B gain %.3f %% share %.3f leg %.3f %%"
              "   C gain %.3f %% at M=%d leg %.3f %%"
              % (name, row["mean_width"], row["weight_renormalised"],
                 row["wide_qmv_gain_pct"], row["leg_share_interpolated"],
                 row["leg_gain_pct"], c_row["wide_qmv_gain_pct"],
                 c_row["gain_read_at_integer_width"], c_row["leg_gain_pct"]))
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
