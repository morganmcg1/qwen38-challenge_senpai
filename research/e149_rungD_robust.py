#!/usr/bin/env python3
"""E149 arm D robustness: is the H-null / H-alt preference carried by one cell?

The free-eight inversion in `research/e149_rungD_curve.py` is inadequate; its
negative control fails at the empirical noise floor and its 5->6 marginal
misses both reference curves. This script asks the smaller, well-conditioned
question that survives that failure: given each named curve its own best
scalar level, which one do the ranked receipts prefer, and does the preference
survive dropping any one prompt?

harness = ranked for every receipt quantity. The H-alt curve is edward's
LOCAL measured curve expressed in ranked units through k = 2.1034, and it is
labelled that way wherever it appears.

Inputs are the design matrix, observations and per-row sigmas already written
by the solver, so this adds no new modelling assumption.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

ORDER = ["beagle", "botany", "drama", "essays", "medicine", "plutarch",
         "republic", "travel"]


def level_fit(mass, y, sigma, base):
    pred = mass @ base
    w = 1.0 / sigma ** 2
    scale = float(np.sum(w * pred * y) / np.sum(w * pred * pred))
    resid = scale * pred - y
    return {"scale": scale,
            "chi2": float(np.sum((resid / sigma) ** 2)),
            "rms_pct": float(np.sqrt(np.mean((resid / y) ** 2)) * 100.0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parent
                    / "e149-rungD.json")
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parent
                    / "e149-rungD-robust.json")
    args = ap.parse_args()

    report = json.loads(args.json.read_text())
    out = {"harness": "ranked",
           "h_alt_note": "edward's LOCAL measured curve divided by k=2.1034, "
                         "expressed in ranked units",
           "models": {}}

    for model in ("tilt", "replay"):
        entry = report["results"][model]
        mass = np.array(entry["mass"])
        y = np.array(entry["y_us_per_round"])
        sigma = np.array(entry["sigma_us_per_round"])
        labels = entry["labels"]
        width = mass.shape[1]
        null = np.array([entry["h_null_truth_used"][str(w + 1)]
                         for w in range(width)])
        alt = np.array([entry["h_alt_truth_used"][str(w + 1)]
                        for w in range(width)])

        rec = {"full": {}, "leave_one_prompt_out": {}, "subsets": {}}
        fn = level_fit(mass, y, sigma, null)
        fa = level_fit(mass, y, sigma, alt)
        rec["full"] = {"h_null": fn, "h_alt": fa,
                       "delta_chi2_null_minus_alt": fn["chi2"] - fa["chi2"],
                       "prefers": "h_null" if fn["chi2"] < fa["chi2"]
                                  else "h_alt"}

        for drop in ORDER:
            keep = [i for i, lab in enumerate(labels) if drop not in lab]
            gn = level_fit(mass[keep], y[keep], sigma[keep], null)
            ga = level_fit(mass[keep], y[keep], sigma[keep], alt)
            rec["leave_one_prompt_out"][drop] = {
                "n_rows": len(keep),
                "delta_chi2_null_minus_alt": gn["chi2"] - ga["chi2"],
                "rms_null_pct": gn["rms_pct"], "rms_alt_pct": ga["rms_pct"],
                "prefers": "h_null" if gn["chi2"] < ga["chi2"] else "h_alt"}

        # The 6/7/8 question lives in the rows that actually carry mass there.
        heavy = float(np.sum(mass[:, 5:], axis=1).max()) * 0.25
        for name, sel in (
            ("width_ge_6_mass_top_half",
             [i for i in range(len(y))
              if mass[i, 5:].sum() >= np.median(mass[:, 5:].sum(axis=1))]),
            ("width_ge_6_mass_over_quarter_of_max",
             [i for i in range(len(y)) if mass[i, 5:].sum() >= heavy]),
            ("drop_plutarch_and_drama",
             [i for i, lab in enumerate(labels)
              if "plutarch" not in lab and "drama" not in lab]),
        ):
            if len(sel) < 3:
                continue
            gn = level_fit(mass[sel], y[sel], sigma[sel], null)
            ga = level_fit(mass[sel], y[sel], sigma[sel], alt)
            rec["subsets"][name] = {
                "rows": [labels[i] for i in sel],
                "delta_chi2_null_minus_alt": gn["chi2"] - ga["chi2"],
                "rms_null_pct": gn["rms_pct"], "rms_alt_pct": ga["rms_pct"],
                "prefers": "h_null" if gn["chi2"] < ga["chi2"] else "h_alt"}

        # Isolate the high-width block. A hybrid keeps one curve's low widths
        # and splices the OTHER curve's step marginals from width 5 upward, so
        # the comparison prices the 5->6, 6->7 and 7->8 cells alone and never
        # the level or the low-width shape.
        def splice(low: np.ndarray, high: np.ndarray, cut: int = 5):
            out_c = low.copy()
            for w in range(cut, len(low)):
                out_c[w] = out_c[w - 1] + (high[w] - high[w - 1])
            return out_c

        hybrids = {
            "h_null_everywhere": null,
            "h_alt_everywhere": alt,
            "low_alt_high_null": splice(alt, null),
            "low_null_high_alt": splice(null, alt),
        }
        rec["high_width_block"] = {}
        for name, base in hybrids.items():
            g = level_fit(mass, y, sigma, base)
            rec["high_width_block"][name] = {
                "chi2": g["chi2"], "rms_pct": g["rms_pct"],
                "curve_us": {w + 1: float(base[w]) for w in range(width)}}
        best = min(rec["high_width_block"],
                   key=lambda k: rec["high_width_block"][k]["chi2"])
        rec["high_width_block_best"] = best
        rec["high_width_block_verdict"] = (
            "replayed marginals at 6,7,8"
            if best in ("h_null_everywhere", "low_alt_high_null")
            else "measured-local marginals at 6,7,8")

        flips = [k for k, v in rec["leave_one_prompt_out"].items()
                 if v["prefers"] != rec["full"]["prefers"]]
        flips += [k for k, v in rec["subsets"].items()
                  if v["prefers"] != rec["full"]["prefers"]]
        rec["preference_flips_on"] = flips
        rec["preference_is_robust"] = not flips
        out["models"][model] = rec

    out["e149_rungD_shape_preference_ranked"] = (
        out["models"]["tilt"]["full"]["prefers"])
    out["e149_rungD_shape_preference_robust"] = bool(
        out["models"]["tilt"]["preference_is_robust"]
        and out["models"]["replay"]["preference_is_robust"]
        and out["models"]["tilt"]["full"]["prefers"]
        == out["models"]["replay"]["full"]["prefers"])
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True))
    print("wrote %s" % args.out)

    for model, rec in out["models"].items():
        print("\n=== %s  full: prefers %s  dchi2=%+.1f  rms null %.3f%% "
              "alt %.3f%%" % (model, rec["full"]["prefers"],
                              rec["full"]["delta_chi2_null_minus_alt"],
                              rec["full"]["h_null"]["rms_pct"],
                              rec["full"]["h_alt"]["rms_pct"]))
        for drop, v in rec["leave_one_prompt_out"].items():
            print("  drop %-9s n=%2d  dchi2=%+10.1f  rms null %6.3f%% "
                  "alt %6.3f%%  prefers %s"
                  % (drop, v["n_rows"], v["delta_chi2_null_minus_alt"],
                     v["rms_null_pct"], v["rms_alt_pct"], v["prefers"]))
        for name, v in rec["subsets"].items():
            print("  %-38s n=%2d dchi2=%+10.1f prefers %s"
                  % (name, len(v["rows"]), v["delta_chi2_null_minus_alt"],
                     v["prefers"]))
        print("  high-width block, low widths held fixed:")
        for name, v in rec["high_width_block"].items():
            print("    %-22s chi2=%12.1f rms=%6.3f%%" % (name, v["chi2"],
                                                         v["rms_pct"]))
        print("    best=%s -> %s" % (rec["high_width_block_best"],
                                     rec["high_width_block_verdict"]))
        print("  robust=%s flips_on=%s" % (rec["preference_is_robust"],
                                           rec["preference_flips_on"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
