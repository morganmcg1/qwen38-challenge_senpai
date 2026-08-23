#!/usr/bin/env python3
"""E145 R1: the regime sweep, and the curve's local closure test.

`harness=local`. Zero GPU; reads the legs `research/e145_read.py` collected.

Two things happen here.

1. **The regime sweep.** `ship` against `pb6` under a real 40 C gate, one
   binary, ABBA palindrome, 512 decode tokens. The advisor pre-registered a
   sign flip between `benchfixture` and `beagle_a`; this scores that.

2. **The closure test, which is the stronger result.** R2 measures the round
   cost at a pinned width. R1 runs the shipped scheduler, which produces a
   *mixture* over widths. Mass-weighting the R2 curve by an R1 leg's own
   realised width histogram predicts that leg's round cost with no free
   parameter. This is the same operation R0 performs in the ranked frame, but
   every input is measured on this host, so it tests the curve rather than the
   replay.

   Advisor Error 156 is the control: interpolating the curve at the *mean*
   width is not an approximation of this, and the two are reported side by side
   so the size of the Jensen gap is visible.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parent.parent

TOKENS = 512

# The advisor's pre-registration in the PR body, and mine in interim 1, as
# percentage change in candidate seconds per token for `pb6` against `ship`.
PREREG = {
    "benchfixture": {"advisor": (-3.0, -1.5), "student": (-3.2, -1.2),
                     "mass_shift_pp": (25.0, None)},
    "beagle_a": {"advisor": (1.0, 3.5), "student": (0.5, 2.5),
                 "mass_shift_pp": (5.0, 20.0)},
    "essays_montaigne": {"advisor": (-0.5, 0.5), "student": (-0.5, 0.8),
                         "mass_shift_pp": (None, 8.0)},
}


def pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b


def verdict(value: float, lo: float | None, hi: float | None) -> str:
    if lo is not None and value < lo:
        return "MISS low"
    if hi is not None and value > hi:
        return "MISS high"
    return "HIT"


def measured_curve(legs: list[dict]) -> dict[int, dict]:
    """Blocks-only microseconds per round at each pinned width, from R2."""
    by_width: dict[int, list[float]] = {}
    for leg in legs:
        if not leg["slot"].startswith("r2-") or leg["pin"] in (None, ""):
            continue
        if not leg["timing_valid"]:
            continue
        by_width.setdefault(int(leg["pin"]) + 1, []).append(
            leg["round_us_from_blocks"])
    return {w: {"us": statistics.fmean(v), "legs": len(v),
                "spread_pct": (pct(max(v), min(v)) if len(v) > 1 else 0.0)}
            for w, v in sorted(by_width.items())}


def curve_at_mean(curve: dict[int, float], width: float) -> float:
    """Advisor Error 156's operation, kept only as a control."""
    lo = int(width)
    lo = max(min(curve), min(lo, max(curve) - 1))
    frac = width - lo
    return curve[lo] + frac * (curve[lo + 1] - curve[lo])


def closure(leg: dict, curve: dict[int, float]) -> dict | None:
    """Predict one shipped leg's round cost from its own width histogram."""
    hist = {int(w): m for w, m in leg["width_hist"].items()}
    covered = sum(m for w, m in hist.items() if w in curve)
    if covered < 0.999:
        return {"covered_mass": covered, "uncovered": sorted(
            w for w in hist if w not in curve)}
    mean_w = sum(w * m for w, m in hist.items()) / sum(hist.values())
    pred = sum(m * curve[w] for w, m in hist.items()) / covered
    obs = leg["round_us_from_blocks"]
    interp = curve_at_mean(curve, mean_w)
    return {
        "covered_mass": covered,
        "mean_width": mean_w,
        "predicted_us": pred,
        "observed_us": obs,
        "error_pct": pct(pred, obs),
        "interp_at_mean_us": interp,
        "interp_error_pct": pct(interp, obs),
        "jensen_gap_us": pred - interp,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", default="research/e145-artifacts/legs.json")
    ap.add_argument("--json", default="research/e145-artifacts/r1.json")
    args = ap.parse_args()

    blob = json.loads((ROOT / args.legs).read_text())
    legs = blob["legs"]
    curve_full = measured_curve(legs)
    curve = {w: v["us"] for w, v in curve_full.items()}

    print("R2 measured local curve, blocks-only us per round")
    for w, v in curve_full.items():
        print(f"  M={w}  {v['us']:>10.1f}   legs {v['legs']}"
              f"   repeat spread {v['spread_pct']:+.3f} %")

    fixtures = sorted({l["fixture"] for l in legs
                       if l["slot"].startswith("r1-")})
    out = {"harness": "local", "gpu_used": False,
           "measured_curve_us": {str(k): v for k, v in curve_full.items()},
           "fixtures": {}}

    for fx in fixtures:
        arms: dict[str, list[dict]] = {}
        for leg in legs:
            if leg["fixture"] != fx or not leg["slot"].startswith("r1-"):
                continue
            arms.setdefault(leg["arm"], []).append(leg)
        entry: dict = {"arms": {}}
        for arm, group in sorted(arms.items()):
            group.sort(key=lambda l: int(l["position"]))
            spt = [l["spt"] for l in group]
            blk = [l["spt_blocks_only"] for l in group]
            rounds = sorted({l["round_count"] for l in group})
            # Round cost must use each leg's own round count before averaging,
            # because the two arms do not run the same number of rounds.
            rc = [l["round_us_from_blocks"] for l in group]
            entry["arms"][arm] = {
                "legs": len(group),
                "positions": [int(l["position"]) for l in group],
                "spt_mean": statistics.fmean(spt),
                "spt_sd": statistics.pstdev(spt),
                "spt_blocks_only_mean": statistics.fmean(blk),
                "round_us_mean": statistics.fmean(rc),
                "round_counts": rounds,
                "mean_draft_len": group[0]["mean_draft_len"],
                "accepted_draft_rate": group[0]["accepted_draft_rate"],
                "rejected_draft_total": group[0]["rejected_draft_total"],
                "replayed_round_count": group[0]["replayed_round_count"],
                "width_hist": group[0]["width_hist"],
                "width_mass_ge6": group[0]["width_mass_ge6"],
                "width_mass_at6": group[0]["width_hist"].get("6", 0.0),
                "entry_temps_c": [l["gate_entry_temp_c"] for l in group],
                "exit_temps_c": [l["leg_exit_temp_c"] for l in group],
                "real_cool_gate_taken": all(
                    l["real_cool_gate_taken"] for l in group),
                "warm_telemetry_present": all(
                    l["warm_telemetry_present"] for l in group),
                "all_tokens_matched": all(
                    l["all_tokens_matched"] for l in group),
                "residual_divergence_count": sum(
                    l["residual_divergence_count"] for l in group),
                "closure": closure(group[0], curve),
            }
        if {"ship", "pb6"} <= set(entry["arms"]):
            s, b = entry["arms"]["ship"], entry["arms"]["pb6"]
            entry["arm_effect_spt_pct"] = pct(b["spt_mean"], s["spt_mean"])
            entry["arm_effect_blocks_pct"] = pct(
                b["spt_blocks_only_mean"], s["spt_blocks_only_mean"])
            entry["arm_effect_round_cost_pct"] = pct(
                b["round_us_mean"], s["round_us_mean"])
            entry["width_mass_ge6_shift_pp"] = 100.0 * (
                s["width_mass_ge6"] - b["width_mass_ge6"])
            entry["width_mass_at6_shift_pp"] = 100.0 * (
                s["width_mass_at6"] - b["width_mass_at6"])
            if s["closure"] and b["closure"] and "predicted_us" in s["closure"]:
                entry["predicted_arm_effect_round_cost_pct"] = pct(
                    b["closure"]["predicted_us"], s["closure"]["predicted_us"])
                entry["arm_effect_prediction_error_pp"] = (
                    entry["predicted_arm_effect_round_cost_pct"]
                    - entry["arm_effect_round_cost_pct"])
            pre = PREREG.get(fx)
            if pre:
                entry["prereg"] = {
                    "advisor": {"range": pre["advisor"], "verdict": verdict(
                        entry["arm_effect_spt_pct"], *pre["advisor"])},
                    "student": {"range": pre["student"], "verdict": verdict(
                        entry["arm_effect_spt_pct"], *pre["student"])},
                    "mass_shift": {
                        "range_pp": pre["mass_shift_pp"],
                        "verdict": verdict(entry["width_mass_ge6_shift_pp"],
                                           *pre["mass_shift_pp"])},
                }
        out["fixtures"][fx] = entry

    for fx, entry in out["fixtures"].items():
        print(f"\n[R1] {fx}")
        for arm, a in entry["arms"].items():
            print(f"  {arm:<7} spt={a['spt_mean']:.9f} sd={a['spt_sd']:.2e}"
                  f" blk_round={a['round_us_mean']:>9.1f}us"
                  f" R={a['round_counts']} dlen={a['mean_draft_len']:.4f}"
                  f" acc={a['accepted_draft_rate']:.4f}"
                  f" m6={a['width_mass_at6']:.4f}"
                  f" ge6={a['width_mass_ge6']:.4f}")
            c = a["closure"]
            if c and "predicted_us" in c:
                print(f"          closure: mass-weighted"
                      f" {c['predicted_us']:>9.1f} vs observed"
                      f" {c['observed_us']:>9.1f}"
                      f"  err {c['error_pct']:+.3f} %"
                      f"   | interp-at-mean {c['interp_error_pct']:+.3f} %"
                      f"  Jensen {c['jensen_gap_us']:+.0f} us")
        if "arm_effect_spt_pct" in entry:
            print(f"  pb6 vs ship: spt {entry['arm_effect_spt_pct']:+.4f} %"
                  f"   blocks {entry['arm_effect_blocks_pct']:+.4f} %"
                  f"   round cost {entry['arm_effect_round_cost_pct']:+.4f} %")
            if "predicted_arm_effect_round_cost_pct" in entry:
                print(f"  curve predicts round cost"
                      f" {entry['predicted_arm_effect_round_cost_pct']:+.4f} %"
                      f"   error"
                      f" {entry['arm_effect_prediction_error_pp']:+.4f} pp")
            print(f"  width mass at 6:"
                  f" {entry['width_mass_at6_shift_pp']:+.2f} pp"
                  f"   at 6 and above:"
                  f" {entry['width_mass_ge6_shift_pp']:+.2f} pp")
            pre = entry.get("prereg")
            if pre:
                for who in ("advisor", "student"):
                    print(f"    prereg {who:<8}"
                          f" {pre[who]['range']} -> {pre[who]['verdict']}")
                print(f"    prereg mass     {pre['mass_shift']['range_pp']}"
                      f" -> {pre['mass_shift']['verdict']}")

    dest = ROOT / args.json
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\ne145_r1 -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
