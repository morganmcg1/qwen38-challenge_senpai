#!/usr/bin/env python3
"""E49 analysis: apply the pre-registered rules to the measured legs.

Reads every leg manifest under `.mlxfast-private/e49-legs/*-leg.json`, pairs it
with that leg's `vendored.json`, and scores the two arms against
`research/e49-prereg.json`. No constant is re-derived here.

  python3 research/e49_analyze.py --out research/e49-artifacts/e49-metrics.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from e46_analyze import bitwise_bad, dispatch_at, jit_spread, load, t_of_m  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
LEGS = REPO / ".mlxfast-private/e49-legs"
PREREG = REPO / "research/e49-prereg.json"


def leg_records() -> list[dict]:
    out = []
    for path in sorted(glob.glob(str(LEGS / "*-leg.json"))):
        leg = json.loads(pathlib.Path(path).read_text())
        tag = leg["tag"]
        try:
            data, ident = load(tag)
        except (OSError, json.JSONDecodeError) as exc:
            out.append({"tag": tag, "arm": leg["arm"], "status": "unreadable",
                        "error": str(exc)})
            continue
        total, per_shape = t_of_m(data)
        out.append({
            "tag": tag,
            "arm": leg["arm"],
            "status": "ok",
            "order": os.path.getmtime(path),
            "t_ms": {m: round(total[m] * 1e3, 3) for m in sorted(total)},
            "per_shape_seconds": {m: per_shape[m] for m in sorted(per_shape)},
            "dispatch": {m: dispatch_at(data, m) for m in sorted(total)},
            "jit_spread_pct": {m: round(jit_spread(data, m), 2) for m in sorted(total)},
            "bitwise_failures": bitwise_bad(data),
            "identity": ident,
            "sources_as_measured": leg["sources_as_measured"],
            "gpu_gate": leg["gpu_gate"],
        })
    out.sort(key=lambda r: r.get("order", 0))
    return out


def by_arm(legs: list[dict]) -> dict[str, list[dict]]:
    arms: dict[str, list[dict]] = {}
    for leg in legs:
        if leg["status"] == "ok":
            arms.setdefault(leg["arm"], []).append(leg)
    return arms


def arm_mean(legs: list[dict], m: int) -> float | None:
    vals = [leg["t_ms"][m] for leg in legs if m in leg["t_ms"]]
    return sum(vals) / len(vals) if vals else None


def arm_spread(legs: list[dict], m: int) -> float:
    vals = [leg["t_ms"][m] for leg in legs if m in leg["t_ms"]]
    return max(vals) - min(vals) if len(vals) > 1 else float("nan")


def contrast(treated: list[dict], control: list[dict], widths: list[int]) -> dict:
    """Per-width delta with its own same-build replicate floor (E46's MDE)."""
    rows = {}
    for m in widths:
        t, c = arm_mean(treated, m), arm_mean(control, m)
        if t is None or c is None:
            continue
        mde = max(arm_spread(treated, m), arm_spread(control, m))
        rows[m] = {
            "control_ms": round(c, 3),
            "treated_ms": round(t, 3),
            "delta_ms": round(t - c, 3),
            "delta_pct": round(100.0 * (t / c - 1.0), 3),
            "mde_ms": round(mde, 3) if mde == mde else None,
            "exceeds_mde": bool(mde == mde and abs(t - c) > mde),
        }
    return rows


def verdict_arm1(rows: dict, prereg: dict) -> dict:
    if 9 not in rows:
        return {"status": "missing", "note": "no M=9 measurement"}
    d = rows[9]["delta_pct"]
    controls = {m: r["delta_pct"] for m, r in rows.items() if m != 9}
    bar = max((abs(v) for v in controls.values()), default=0.0)
    # Every scored width dispatches through the same `affine_qmv_fast`
    # [[kernel]], so the control widths are exposed to iso5's raised allocation
    # while running byte-identical code. Their pooled movement is therefore a
    # shared-tax readout taken inside arm 1, and subtracting it from M=9 leaves
    # the local effect. Reported only; the pre-registered rule reads raw d.
    pooled_ctrl = sum(controls.values()) / len(controls) if controls else None
    rules = prereg["decision_rules"]
    if d >= -2.0:
        label, rule = "H_local_eaten", rules["arm1_eaten"]
    elif d <= -8.0:
        label, rule = "H_local_win", rules["arm1_local_win"]
    else:
        label, rule = "partial_local_win", rules["arm1_middle"]
    return {
        "delta_pct_m9": d,
        "delta_ms_m9": rows[9]["delta_ms"],
        "mde_ms_m9": rows[9]["mde_ms"],
        "exceeds_mde": rows[9]["exceeds_mde"],
        "predicted_pct": [prereg["arm1_isolated"]["predicted_pct_from_e46_refit"],
                          prereg["arm1_isolated"]["predicted_pct_from_contrast_b"]],
        "unchanged_code_widths_pct": controls,
        "worst_unchanged_code_width_pct": round(bar, 3),
        "exceeds_unchanged_code_bar": bool(abs(d) > bar),
        "pooled_unchanged_code_pct": (
            round(pooled_ctrl, 3) if pooled_ctrl is not None else None),
        "local_effect_pct_tax_corrected": (
            round(d - pooled_ctrl, 3) if pooled_ctrl is not None else None),
        "tax_corrected_is_reported_not_rule_input": True,
        "label": label,
        "rule_fired": rule,
        "caveat": "the unchanged-code widths run the generic path in BOTH "
                  "isolated builds but SHARE the kernel allocation, so their "
                  "movement is not a pure noise bar: it is itself a shared-tax "
                  "readout. The same-build replicate spread is the noise bar.",
    }


def verdict_arm2(arms: dict, prereg: dict, untouched: list[int]) -> dict:
    if "shipped" not in arms:
        return {"status": "missing", "note": "no control arm measured"}
    out = {"untouched_widths": untouched, "doses": {}}
    for name in ("dose_null", "dose_129", "dose_big", "dose_huge", "e27_replica"):
        if name not in arms:
            continue
        widths = untouched if name != "e27_replica" else [m for m in untouched if m != 9]
        rows = contrast(arms[name], arms["shipped"], widths)
        if not rows:
            continue
        pooled = sum(r["delta_pct"] for r in rows.values()) / len(rows)
        worst = max(rows.values(), key=lambda r: abs(r["delta_pct"]))
        out["doses"][name] = {
            "per_width": rows,
            "pooled_tax_pct": round(pooled, 3),
            "worst_width_pct": worst["delta_pct"],
            "n_widths_slower": sum(1 for r in rows.values() if r["delta_ms"] > 0),
            "n_widths": len(rows),
        }
        if name == "e27_replica" and 9 in arms[name][0]["t_ms"]:
            m9 = contrast(arms[name], arms["shipped"], [9])
            out["doses"][name]["m9_composite"] = m9.get(9)

    rules = prereg["decision_rules"]
    null = out["doses"].get("dose_null", {}).get("pooled_tax_pct")
    at129 = out["doses"].get("dose_129", {}).get("pooled_tax_pct")
    if null is not None and abs(null) > 2.0:
        out["label"], out["rule_fired"] = "ladder_invalid", rules["arm2_invalid"]
    elif at129 is None:
        out["label"] = "incomplete"
    elif at129 >= 8.0:
        out["label"], out["rule_fired"] = "tax_confirmed", rules["arm2_confirmed"]
    elif abs(at129) <= 2.0:
        out["label"], out["rule_fired"] = "tax_refuted", rules["arm2_refuted"]
    else:
        out["label"] = "ambiguous_middle"
    out["dose_response"] = {
        name: out["doses"][name]["pooled_tax_pct"]
        for name in ("dose_null", "dose_129", "dose_big", "dose_huge")
        if name in out["doses"]
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e49-artifacts/e49-metrics.json")
    args = ap.parse_args()

    prereg = json.loads(PREREG.read_text())
    legs = leg_records()
    arms = by_arm(legs)

    print("E49 legs, in run order:")
    print("  %-22s %-13s %-9s %s" % ("tag", "arm", "gate", "T(M) ms"))
    for leg in legs:
        if leg["status"] != "ok":
            print("  %-22s %-13s %s" % (leg["tag"], leg["arm"], leg["status"]))
            continue
        ts = " ".join("%d:%.1f" % (m, v) for m, v in sorted(leg["t_ms"].items()))
        print("  %-22s %-13s %-9s %s"
              % (leg["tag"], leg["arm"], leg["gpu_gate"]["state"], ts))

    payload = {"legs": legs, "arms_measured": sorted(arms)}

    if {"iso3", "iso5"} <= set(arms):
        widths = sorted(set(arms["iso3"][0]["t_ms"]) & set(arms["iso5"][0]["t_ms"]))
        rows = contrast(arms["iso5"], arms["iso3"], widths)
        v = verdict_arm1(rows, prereg)
        payload["arm1"] = {"per_width": rows, "verdict": v}
        print("\nARM 1 -- isolated <T,9,5> vs <T,9,3>")
        print("  %-4s %-11s %-11s %-10s %-9s %-8s" %
              ("M", "iso3 ms", "iso5 ms", "delta ms", "delta %", "MDE ms"))
        for m, r in sorted(rows.items()):
            print("  %-4d %-11.3f %-11.3f %-10.3f %-9.3f %-8s %s"
                  % (m, r["control_ms"], r["treated_ms"], r["delta_ms"],
                     r["delta_pct"], r["mde_ms"],
                     "<- contrast" if m == 9 else "unchanged code"))
        print("  predicted %s %%   measured %+.2f %%   ->  %s"
              % (v["predicted_pct"], v["delta_pct_m9"], v["label"]))
        if v.get("pooled_unchanged_code_pct") is not None:
            print("  shared-allocation readout from the unchanged widths "
                  "%+.2f %%  =>  local effect %+.2f %% (reported, not the rule)"
                  % (v["pooled_unchanged_code_pct"],
                     v["local_effect_pct_tax_corrected"]))

    if "shipped" in arms:
        untouched = prereg["arm2_shared"]["untouched_widths"]
        v = verdict_arm2(arms, prereg, untouched)
        payload["arm2"] = v
        print("\nARM 2 -- ceiling dose ladder on untouched widths")
        for name, d in v.get("doses", {}).items():
            print("  %-13s pooled %+7.3f %%   worst %+7.3f %%   %d/%d slower"
                  % (name, d["pooled_tax_pct"], d["worst_width_pct"],
                     d["n_widths_slower"], d["n_widths"]))
        print("  verdict: %s" % v.get("label"))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
