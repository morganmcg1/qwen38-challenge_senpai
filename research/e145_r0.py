#!/usr/bin/env python3
"""E145 R0: falsify the rebuilt width curve against ranked round costs.

`harness=ranked frame, replayed`. Zero GPU.

FINDING 219 gives a per-prompt ranked round cost in absolute microseconds:

    tokens per round = 1 + a * d          (8 of 8, worst error 0.33 rounds)
    round cost       = seconds per token * 512 / R

The E134 item 2 replay gives a realised verify-width histogram per prompt.
Together they predict that same round cost:

    predicted(prompt) = SUM over w of mass[prompt][w] * round_us[w]

WHAT THE TEST CAN AND CANNOT SHOW. Three things separate the prediction from
the F219 column, and only the third is a fault in the curve:

  1. ANCHOR RECEIPT. The curve and the histogram were both produced at receipt
     `623e77af`. F219 measures `1760479a`. The candidate got faster between
     them, so a uniform level offset is EXPECTED and is not curve error. The
     artifact carries `623e77af`'s own measured per-prompt round costs, so the
     in-sample check is run first and reported separately.
  2. OPERATING POINT. The histogram is the scheduler's behaviour at
     `623e77af`. If its implied mean draft length differs from the draft length
     F219 measured, the prediction is being asked about a different operating
     point. That is checked per prompt and reported, not assumed away.
  3. SHAPE. After one scalar level transfer is removed, the residual per prompt
     is the disagreement that no receipt or operating-point difference can
     explain. THAT is the number that says whether R2 repairs something
     load-bearing.

Reporting only the raw signed error against F219 would charge the curve for
items 1 and 2, which is the same mistake as reading a level from a step.

Usage:
  python3 research/e145_r0.py --json research/e145-artifacts/r0.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parent.parent
CURVE_JSON = ROOT / "research/e134-artifacts/item2-measured-curve.json"

# FINDING 219, receipt `1760479a`, advisor supplied. `round_us` is recomputed
# here from `cand_spt`, `R` and the 512-token window rather than copied, so a
# transcription error in either source shows up as a disagreement.
F219 = {
    "beagle":   {"cand_spt": 0.0106863, "dlen": 4.3818, "R": 110.00},
    "botany":   {"cand_spt": 0.0096534, "dlen": 6.1481, "R": 81.04},
    "drama":    {"cand_spt": 0.0178217, "dlen": 2.2976, "R": 252.01},
    "essays":   {"cand_spt": 0.0098320, "dlen": 5.0870, "R": 92.04},
    "medicine": {"cand_spt": 0.0097197, "dlen": 5.2556, "R": 90.01},
    "plutarch": {"cand_spt": 0.0300828, "dlen": 0.1557, "R": 486.76},
    "republic": {"cand_spt": 0.0097098, "dlen": 4.9892, "R": 93.00},
    "travel":   {"cand_spt": 0.0156178, "dlen": 2.6479, "R": 212.33},
}
F219_QUOTED_ROUND_US = {
    "beagle": 49738.5, "botany": 60991.4, "drama": 36207.0,
    "essays": 54695.7, "medicine": 55285.4, "plutarch": 31642.9,
    "republic": 53455.4, "travel": 37659.6,
}
TOKENS = 512

# The prompts that can occupy the median pair. Rule 121 reads positions 3 and
# 4 of the sorted eight, and drama, travel and plutarch never reach them.
MEDIAN_ELIGIBLE = ("beagle", "botany", "essays", "medicine", "republic")

# FINDING 218 / F2 section 3. The identified bound on beagle's ranked
# round-cost rise for `09b452f3` against `1760479a`.
F218_SHIFT = 0.0953
F218_BOUND_US = (119.0, 970.0)


# F4 / Advisor Error 156. `684821ed` and `1760479a` are the same candidate:
# their draft lengths are digit-identical on all eight prompts, so the width
# histogram, and therefore any mass-weighted prediction, is identical for both.
# The two rows' round costs must then differ only by receipt noise, which makes
# their gap the noise floor that every R0 residual has to clear before it can
# be read as curve error.
NULL_PAIR_RECEIPT = "684821ed"
BOARD_CACHE = pathlib.Path("/tmp/yukon-board/full.json")
PROMPT_BY_HASH = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}


def pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b


def null_control(f219_round_us: dict[str, float]) -> dict:
    """Measure how tightly two receipts of the same candidate agree.

    Returns `{}` when the board cache is absent, because R0's own result does
    not depend on it; the control only calibrates how to read the residuals.
    """
    if not BOARD_CACHE.exists():
        return {}
    payload = json.loads(BOARD_CACHE.read_text())
    rows = payload if isinstance(payload, list) else next(
        payload[k] for k in ("submissions", "data", "items", "results")
        if isinstance(payload.get(k), list))
    hit = [r for r in rows
           if str(r.get("id", "")).startswith(NULL_PAIR_RECEIPT)]
    if len(hit) != 1:
        return {}
    per_prompt = {PROMPT_BY_HASH[e["prompt_sha256"][:8]]: e
                  for e in hit[0]["officialMetrics"]["per_prompt"]}

    out = {"receipt": NULL_PAIR_RECEIPT, "vs": "1760479a", "prompts": {}}
    for p, ref in F219.items():
        entry = per_prompt[p]
        dlen = entry["effective_mean_draft_len"]
        spt = entry["mtp_seconds_per_token_mean"]
        # Draft lengths are identical, so tokens per round is identical and the
        # round count carries over unchanged from F219.
        round_us = spt * 1e6 * TOKENS / ref["R"]
        out["prompts"][p] = {
            "draft_len": dlen,
            "draft_len_gap_vs_f219": dlen - ref["dlen"],
            "round_us": round_us,
            "round_us_pct_vs_f219": pct(round_us, f219_round_us[p]),
        }
    eligible = [abs(out["prompts"][p]["round_us_pct_vs_f219"])
                for p in MEDIAN_ELIGIBLE]
    gaps = [abs(out["prompts"][p]["draft_len_gap_vs_f219"]) for p in F219]
    out["max_abs_draft_len_gap"] = max(gaps)
    out["is_true_null_pair"] = max(gaps) < 1e-3
    out["noise_floor_pct_median_eligible"] = max(eligible)
    out["noise_floor_pct_all"] = max(
        abs(v["round_us_pct_vs_f219"]) for v in out["prompts"].values())
    return out


def load() -> tuple[dict, dict, dict]:
    blob = json.loads(CURVE_JSON.read_text())
    forms = {name: {w: us for w, us in
                    zip(entry["shape"]["rows"], entry["shape"]["round_us"])}
             for name, entry in blob["curves"].items()}
    masses = {p: {int(w): m for w, m in entry["by_width"].items()}
              for p, entry in blob["width_masses"].items()}
    return blob, forms, masses


def mean_width(hist: dict[int, float]) -> float:
    total = sum(hist.values())
    return sum(w * m for w, m in hist.items()) / total


def predict(hist: dict[int, float], curve: dict[int, float]) -> float:
    total = sum(hist.values())
    return sum(m * curve[w] for w, m in hist.items()) / total


def fit_level(pred: dict[str, float], obs: dict[str, float],
              prompts) -> dict:
    """One scalar `k` with `obs ~= k * pred`, fitted in log space."""
    logs = [math.log(obs[p] / pred[p]) for p in prompts]
    k = math.exp(statistics.fmean(logs))
    residual = {p: pct(k * pred[p], obs[p]) for p in prompts}
    signs = {p: (1 if residual[p] > 0 else -1) for p in prompts}
    return {
        "k": k,
        "residual_pct": residual,
        "max_abs_residual_pct": max(abs(v) for v in residual.values()),
        "systematic_sign": abs(sum(signs.values())) == len(prompts),
        "positive_residuals": sum(1 for v in residual.values() if v > 0),
    }


def shift_hist(hist: dict[int, float], delta: float,
               top: int | None = None) -> tuple[dict[int, float], float]:
    """Raise the mean realised width by exactly `delta`.

    A fraction `theta` of every width's mass moves up one width, which is the
    shift an across-the-board acceptance improvement produces under a threshold
    walk: every round becomes `theta` more likely to clear its next threshold.
    Mass already at the top width cannot move, because the shipped envelope
    caps it, so the mean rises by `theta * (1 - mass_top)` and `theta` must be
    solved for rather than set equal to `delta`. Getting this wrong understates
    the predicted cost by the reciprocal of the mobile mass, which on beagle is
    a factor of 1.5.
    """
    top = max(hist) if top is None else top
    total = sum(hist.values())
    mobile = 1.0 - sum(m for w, m in hist.items() if w >= top) / total
    if mobile <= 0.0:
        return dict(hist), 0.0
    theta = delta / mobile
    out: dict[int, float] = {}
    for w, mass in hist.items():
        if w >= top:
            out[w] = out.get(w, 0.0) + mass
            continue
        out[w] = out.get(w, 0.0) + mass * (1.0 - theta)
        out[w + 1] = out.get(w + 1, 0.0) + mass * theta
    return out, theta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="research/e145-artifacts/r0.json")
    args = ap.parse_args()

    blob, forms, masses = load()
    best = blob["best_form"]
    anchor_receipt = blob["receipt"]
    anchor_table = blob["measured_table"]

    f219_round_us = {p: v["cand_spt"] * 1e6 * TOKENS / v["R"]
                     for p, v in F219.items()}
    quote_error = {p: pct(f219_round_us[p], F219_QUOTED_ROUND_US[p])
                   for p in F219}
    null = null_control(f219_round_us)

    prompts = sorted(F219)
    out_prompts = {}
    for p in prompts:
        hist = masses[p]
        full = {w: hist.get(w, 0.0) for w in range(1, 10)}
        implied_dlen = mean_width(hist) - 1.0
        out_prompts[p] = {
            "width_hist": full,
            "hist_mass_total": sum(full.values()),
            "hist_rounds": blob["width_masses"][p]["rounds"],
            "hist_drafting_share": blob["width_masses"][p]["drafting_share"],
            "implied_draft_len": implied_dlen,
            "ranked_draft_len_1760479a": F219[p]["dlen"],
            "draft_len_gap": implied_dlen - F219[p]["dlen"],
            "f219_round_us": f219_round_us[p],
            "f219_quoted_round_us": F219_QUOTED_ROUND_US[p],
            "f219_quote_reproduction_pct": quote_error[p],
            "anchor_round_us_before": anchor_table[p]["before"],
            "anchor_round_us_after": anchor_table[p]["after"],
            "predicted_round_us": {},
            "vs_f219_pct": {},
            "vs_anchor_after_pct": {},
        }
        for name, curve in forms.items():
            pred = predict(hist, curve)
            out_prompts[p]["predicted_round_us"][name] = pred
            out_prompts[p]["vs_f219_pct"][name] = pct(pred, f219_round_us[p])
            out_prompts[p]["vs_anchor_after_pct"][name] = pct(
                pred, anchor_table[p]["after"])

    summary = {}
    for name in forms:
        pred = {p: out_prompts[p]["predicted_round_us"][name] for p in prompts}
        vs_f219 = {p: out_prompts[p]["vs_f219_pct"][name] for p in prompts}
        vs_anchor = {p: out_prompts[p]["vs_anchor_after_pct"][name]
                     for p in prompts}
        weights = {p: f219_round_us[p] for p in MEDIAN_ELIGIBLE}
        wmae = (sum(weights[p] * abs(vs_f219[p]) for p in MEDIAN_ELIGIBLE)
                / sum(weights.values()))
        summary[name] = {
            "weighted_mae_pct_median_eligible": wmae,
            "mean_signed_pct_all": statistics.fmean(vs_f219.values()),
            "mean_signed_pct_median_eligible": statistics.fmean(
                [vs_f219[p] for p in MEDIAN_ELIGIBLE]),
            "systematic_sign_all": (
                all(v > 0 for v in vs_f219.values())
                or all(v < 0 for v in vs_f219.values())),
            "vs_anchor_mean_signed_pct": statistics.fmean(vs_anchor.values()),
            "vs_anchor_max_abs_pct": max(abs(v) for v in vs_anchor.values()),
            "level_fit_all": fit_level(pred, f219_round_us, prompts),
            "level_fit_median_eligible": fit_level(pred, f219_round_us,
                                                   MEDIAN_ELIGIBLE),
            "level_fit_vs_anchor": fit_level(
                pred, {p: anchor_table[p]["after"] for p in prompts}, prompts),
        }

    # F218: price beagle's +0.0953 draft-length shift on the REPLAYED curve,
    # which is already in the ranked frame, so no transfer is needed. R2 will
    # repeat this with the measured curve.
    beagle = masses["beagle"]
    shifted, theta = shift_hist(beagle, F218_SHIFT)
    f218 = {
        "shift_requested": F218_SHIFT,
        "shift_realised": mean_width(shifted) - mean_width(beagle),
        "promotion_probability_theta": theta,
        "mobile_mass": F218_SHIFT / theta if theta else 0.0,
        "bound_us": list(F218_BOUND_US),
        "forms": {},
    }
    for name, curve in forms.items():
        before = predict(beagle, curve)
        after = predict(shifted, curve)
        rise = after - before
        f218["forms"][name] = {
            "round_us_before": before,
            "round_us_after": after,
            "predicted_rise_us": rise,
            "predicted_rise_pct": pct(after, before),
            "inside_bound": F218_BOUND_US[0] <= rise <= F218_BOUND_US[1],
            "above_bound": rise > F218_BOUND_US[1],
            "below_bound": rise < F218_BOUND_US[0],
        }

    result = {
        "harness": "ranked frame, replayed",
        "gpu_used": False,
        "anchor_receipt": anchor_receipt,
        "f219_receipt": "1760479a",
        "best_form": best,
        "median_eligible": list(MEDIAN_ELIGIBLE),
        "curves": {name: {str(w): us for w, us in curve.items()}
                   for name, curve in forms.items()},
        "prompts": out_prompts,
        "summary": summary,
        "f218_beagle_shift": f218,
        "null_control": null,
        "e145_predicted_beagle_shift_us": f218["forms"][best][
            "predicted_rise_us"],
        "e145_f218_bound_consistent": f218["forms"][best]["inside_bound"],
    }
    out = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(f"e145_r0 -> {out}")
    print(f"curve and histogram anchor receipt: {anchor_receipt}")
    print(f"F219 measures receipt:              1760479a")
    print(f"best form: {best}\n")

    print("F219 round cost reproduced from cand_spt * 512 / R")
    for p in prompts:
        print(f"  {p:<9} recomputed {f219_round_us[p]:>9.1f} us"
              f"   quoted {F219_QUOTED_ROUND_US[p]:>9.1f}"
              f"   {quote_error[p]:>+7.3f} %")

    print("\noperating point: does the histogram describe F219's decode?")
    for p in prompts:
        e = out_prompts[p]
        print(f"  {p:<9} hist draft len {e['implied_draft_len']:>7.4f}"
              f"   ranked {e['ranked_draft_len_1760479a']:>7.4f}"
              f"   gap {e['draft_len_gap']:>+7.4f}"
              f"   mass {e['hist_mass_total']:.6f}")

    for name in (best, "per_round"):
        s = summary[name]
        print(f"\n=== form {name} ===")
        print(f"  {'prompt':<9} {'predicted':>10} {'F219':>10} {'err%':>8}"
              f" {'anchor@623e77af':>16} {'err%':>8}")
        for p in prompts:
            e = out_prompts[p]
            print(f"  {p:<9} {e['predicted_round_us'][name]:>10.1f}"
                  f" {e['f219_round_us']:>10.1f}"
                  f" {e['vs_f219_pct'][name]:>+8.2f}"
                  f" {e['anchor_round_us_after']:>16.1f}"
                  f" {e['vs_anchor_after_pct'][name]:>+8.2f}")
        print(f"  weighted MAE against F219, five median-eligible prompts:"
              f" {s['weighted_mae_pct_median_eligible']:.3f} %")
        print(f"  mean signed against F219 {s['mean_signed_pct_all']:+.3f} %"
              f"   systematic sign {s['systematic_sign_all']}")
        print(f"  mean signed against its OWN anchor receipt"
              f" {s['vs_anchor_mean_signed_pct']:+.3f} %"
              f"   worst {s['vs_anchor_max_abs_pct']:.3f} %")
        lf = s["level_fit_all"]
        print(f"  after one scalar level transfer k={lf['k']:.4f}:"
              f" worst shape residual {lf['max_abs_residual_pct']:.3f} %"
              f"   systematic sign {lf['systematic_sign']}")
        for p in prompts:
            print(f"      {p:<9} shape residual"
                  f" {lf['residual_pct'][p]:>+7.3f} %")

    print(f"\nF218 beagle +{F218_SHIFT} draft length, replayed ranked curve")
    print(f"  identified bound {F218_BOUND_US[0]:.0f} to"
          f" {F218_BOUND_US[1]:.0f} us")
    for name in forms:
        r = f218["forms"][name]
        print(f"  {name:<20} rise {r['predicted_rise_us']:>+9.1f} us"
              f"  ({r['predicted_rise_pct']:+.4f} %)"
              f"   inside bound: {r['inside_bound']}")
    print(f"  realised shift {f218['shift_realised']:.4f}"
          f"   theta {f218['promotion_probability_theta']:.5f}"
          f"   mobile mass {f218['mobile_mass']:.4f}")
    print(f"  e145_predicted_beagle_shift_us"
          f" {result['e145_predicted_beagle_shift_us']:.1f}")
    print(f"  e145_f218_bound_consistent"
          f" {result['e145_f218_bound_consistent']}")

    if null:
        print(f"\nF4 null control: {null['receipt']} against {null['vs']},"
              f" same candidate, round cost in us")
        for p in prompts:
            e = null["prompts"][p]
            star = " *" if p in MEDIAN_ELIGIBLE else ""
            print(f"  {p:<9} dlen {e['draft_len']:>8.4f}"
                  f"  gap {e['draft_len_gap_vs_f219']:>+9.2e}"
                  f"  round {e['round_us']:>9.1f}"
                  f"  {e['round_us_pct_vs_f219']:>+7.3f} %{star}")
        print(f"  true null pair: {null['is_true_null_pair']}"
              f"   max abs draft-length gap {null['max_abs_draft_len_gap']:.2e}")
        print(f"  NOISE FLOOR, median-eligible:"
              f" {null['noise_floor_pct_median_eligible']:.3f} pp"
              f"   all eight: {null['noise_floor_pct_all']:.3f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
