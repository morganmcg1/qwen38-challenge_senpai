#!/usr/bin/env python3
"""E54 analysis: apply the pre-registered falsifiers to the measured legs.

Reads every leg manifest under `.mlxfast-private/e54-legs/*-leg.json`, pairs it
with that leg's `vendored.json`, and scores the pairs against
`research/e54-artifacts/e54-prereg.json`. No constant is re-derived here.

  python3 research/e54_analyze.py --out research/e54-artifacts/e54-metrics.json
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
from e49_analyze import contrast  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
LEGS = REPO / ".mlxfast-private/e54-legs"
PREREG = REPO / "research/e54-artifacts/e54-prereg.json"

# pair -> (control arm, treated arm, treated width)
PAIRS = {
    "P1": ("iso_m5_ipg3", "iso_m5_ipg5", 5),
    "P2": ("iso_m7_ipg4", "iso_m7_ipg5", 7),
    "P3": ("iso_m8_ipg4", "iso_m8_ipg5", 8),
    "P4": ("shipped", "e27_full", None),  # composite: widths 5 and 9 both treated
}
P4_TREATED_WIDTHS = (5, 9)


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


def pair_report(arms: dict, pair: str, prereg: dict) -> dict | None:
    control_arm, treated_arm, treated_m = PAIRS[pair]
    if control_arm not in arms or treated_arm not in arms:
        return None
    widths = sorted(set(arms[control_arm][0]["t_ms"])
                    & set(arms[treated_arm][0]["t_ms"]))
    rows = contrast(arms[treated_arm], arms[control_arm], widths)
    if not rows:
        return None

    treated = list(P4_TREATED_WIDTHS) if pair == "P4" else [treated_m]
    controls = {m: r["delta_pct"] for m, r in rows.items() if m not in treated}
    control_bar = max((abs(v) for v in controls.values()), default=0.0)
    pooled_ctrl = sum(controls.values()) / len(controls) if controls else None

    # The pre-registered bar is the largest of the session-noise floor, the
    # worst unchanged-code width, and the treated width's own replicate spread.
    floor = prereg["decision_bar_pct"]
    spreads = []
    for m in treated:
        if m in rows and rows[m]["mde_ms"] is not None and rows[m]["control_ms"]:
            spreads.append(100.0 * rows[m]["mde_ms"] / rows[m]["control_ms"])
    bar = max([floor, control_bar] + spreads)

    per_treated = {}
    for m in treated:
        if m not in rows:
            continue
        d = rows[m]["delta_pct"]
        per_treated[m] = {
            "delta_pct": d,
            "delta_ms": rows[m]["delta_ms"],
            "exceeds_bar": bool(abs(d) > bar),
            "direction": "win" if d < -bar else ("regress" if d > bar else "null"),
            "local_effect_pct_tax_corrected": (
                round(d - pooled_ctrl, 3) if pooled_ctrl is not None else None),
        }

    cell = prereg["cells"].get(pair)
    return {
        "control_arm": control_arm,
        "treated_arm": treated_arm,
        "treated_widths": treated,
        "per_width": rows,
        "treated": per_treated,
        "bar_pct": round(bar, 3),
        "bar_inputs": {
            "prereg_floor_pct": floor,
            "worst_unchanged_code_width_pct": round(control_bar, 3),
            "treated_replicate_spread_pct": [round(s, 3) for s in spreads],
        },
        "unchanged_code_widths_pct": controls,
        "pooled_unchanged_code_pct": (
            round(pooled_ctrl, 3) if pooled_ctrl is not None else None),
        "tax_corrected_is_reported_not_rule_input": True,
        "predicted_pct": cell["pred_pct"] if cell else None,
        "caveat": "the unchanged-code widths run byte-identical code in BOTH "
                  "arms but SHARE one [[kernel]] allocation, so their movement "
                  "is a shared-tax readout, not a pure noise bar",
    }


def law_verdicts(observed: dict[str, float], bar: float) -> dict:
    """Score each pre-registered law against the observed treated-width deltas.

    `observed` maps pair name to its treated-width delta percent. A law is
    falsified only by the falsifier written down before the measurement.
    """
    p1, p2, p3 = (observed.get(k) for k in ("P1", "P2", "P3"))
    have = {k: v for k, v in observed.items() if v is not None}
    out = {}

    def verdict(name: str, reasons: list[str], ready: bool) -> None:
        out[name] = {
            "falsified": bool(reasons),
            "reasons": reasons,
            "decidable": ready,
        }

    if p2 is not None and p3 is not None and p1 is not None:
        reasons = []
        if p2 > -bar:
            reasons.append("P2 did not win by more than the bar (%.3f %%)" % p2)
        if p3 > -bar:
            reasons.append("P3 did not win by more than the bar (%.3f %%)" % p3)
        if p1 >= -12.0:
            reasons.append("P1 did not win at least 12 %% (%.3f %%)" % p1)
        verdict("A_as_written_in_brief", reasons, True)

        reasons = []
        if abs(p1) < bar:
            reasons.append("P1 stayed inside the bar (%.3f %%)" % p1)
        if p2 < -3.0:
            reasons.append("P2 won more than 3 %% with no traffic change (%.3f %%)" % p2)
        if p3 < -3.0:
            reasons.append("P3 won more than 3 %% with no traffic change (%.3f %%)" % p3)
        verdict("A_prime_working_group_traffic", reasons, True)

        reasons = []
        if p1 < -bar:
            reasons.append("P1 won by more than the bar (%.3f %%)" % p1)
        if p2 > -3.0:
            reasons.append("P2 did not win at least 3 %% (%.3f %%)" % p2)
        if p3 > -3.0:
            reasons.append("P3 did not win at least 3 %% (%.3f %%)" % p3)
        verdict("C_sibling_overlap_advisor", reasons, True)

    reasons = [f"{k} won by more than the bar ({v:.3f} %)"
               for k, v in have.items() if v < -bar]
    verdict("B_na5_always_loses", reasons or ["already falsified at M=9 by E49"],
            True)

    reasons = [f"{k} moved past the bar ({v:.3f} %)"
               for k, v in have.items() if abs(v) > bar]
    verdict("null", reasons, bool(have))

    surviving = [k for k, v in out.items() if not v["falsified"] and v["decidable"]]
    out["surviving"] = surviving
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e54-artifacts/e54-metrics.json")
    args = ap.parse_args()

    prereg = json.loads(PREREG.read_text())
    legs = leg_records()
    arms = by_arm(legs)

    print("E54 legs, in run order:")
    print("  %-24s %-26s %-9s %s" % ("tag", "arm", "gate", "T(M) ms"))
    for leg in legs:
        if leg["status"] != "ok":
            print("  %-24s %-26s %s" % (leg["tag"], leg["arm"], leg["status"]))
            continue
        ts = " ".join("%d:%.1f" % (m, v) for m, v in sorted(leg["t_ms"].items()))
        print("  %-24s %-26s %-9s %s"
              % (leg["tag"], leg["arm"], leg["gpu_gate"]["state"], ts))

    payload = {"legs": legs, "arms_measured": sorted(arms), "pairs": {}}
    observed: dict[str, float] = {}
    bars: list[float] = []

    for pair in PAIRS:
        rep = pair_report(arms, pair, prereg)
        if rep is None:
            continue
        payload["pairs"][pair] = rep
        bars.append(rep["bar_pct"])
        if pair != "P4" and rep["treated"]:
            m = rep["treated_widths"][0]
            if m in rep["treated"]:
                observed[pair] = rep["treated"][m]["delta_pct"]

        cell = prereg["cells"].get(pair)
        title = (f"{pair} -- {cell['shipped_template']} vs "
                 f"{cell['candidate_template']}" if cell
                 else f"{pair} -- {rep['control_arm']} vs {rep['treated_arm']}")
        print(f"\n{title}")
        print("  %-4s %-11s %-11s %-10s %-9s %-8s %s"
              % ("M", "control", "treated", "delta ms", "delta %", "MDE ms", ""))
        for m, r in sorted(rep["per_width"].items()):
            mark = "<- TREATED" if m in rep["treated_widths"] else "unchanged code"
            print("  %-4d %-11.3f %-11.3f %-10.3f %-9.3f %-8s %s"
                  % (m, r["control_ms"], r["treated_ms"], r["delta_ms"],
                     r["delta_pct"], r["mde_ms"], mark))
        print("  bar %.3f %%  (floor %.3f, worst unchanged %.3f, spread %s)"
              % (rep["bar_pct"], rep["bar_inputs"]["prereg_floor_pct"],
                 rep["bar_inputs"]["worst_unchanged_code_width_pct"],
                 rep["bar_inputs"]["treated_replicate_spread_pct"]))
        if rep["predicted_pct"]:
            p = rep["predicted_pct"]
            print("  predicted: group-law %+.2f %% (cal %+.2f), bandwidth "
                  "ser %+.2f / ovl %+.2f / theta %+.2f"
                  % (p["working_group_law_raw"],
                     p["working_group_law_calibrated"],
                     p["bandwidth_serialized"], p["bandwidth_overlapped"],
                     p["bandwidth_theta_calibrated"]))
        for m, t in sorted(rep["treated"].items()):
            print("  M=%d measured %+.3f %%  ->  %s"
                  % (m, t["delta_pct"], t["direction"]))
        if rep["pooled_unchanged_code_pct"] is not None:
            print("  shared-allocation readout %+.3f %% (reported, not the rule)"
                  % rep["pooled_unchanged_code_pct"])

    if observed:
        bar = max(bars) if bars else prereg["decision_bar_pct"]
        laws = law_verdicts(observed, bar)
        payload["law_verdicts"] = laws
        payload["observed_treated_pct"] = observed
        payload["bar_used_for_laws_pct"] = round(bar, 3)
        print("\nLAW VERDICTS  (bar %.3f %%)" % bar)
        for name, v in laws.items():
            if name == "surviving":
                continue
            state = "FALSIFIED" if v["falsified"] else "survives"
            print("  %-32s %-10s %s"
                  % (name, state, "; ".join(v["reasons"]) or "-"))
        print("  surviving: %s" % (", ".join(laws["surviving"]) or "none"))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
