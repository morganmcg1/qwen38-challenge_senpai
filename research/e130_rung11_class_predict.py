#!/usr/bin/env python3
"""E130 rung 11, F17 section 3: turn the admission class table into a predictor.

The admission probe is cheap and runs before the timing legs. Until now it was
only a safety instrument. This script makes it produce a falsifiable per-rung
prediction of the timing delta, written down before the expensive measurement
runs.

THE PREDICTION. Rung 10a measured `s64 -> s512`, an admitted delta of 448 MiB,
at -0.1968 % candidate seconds per token. That is -0.2249 % per 512 MiB. If the
value of an admitted byte depends on what kind of state it holds, and the
17,825,792 B per-layer KV class is the hot state, then a rung that admits a
colder mix should return proportionally less:

    predicted_pct = -0.2249
                    * (admitted_MiB_at_this_rung / 512)
                    * (kv_fraction_at_this_rung / kv_fraction_at_s64_to_s512)

At the anchor step both ratios are 1 and 448/512, so the formula reproduces
-0.1968 % exactly. That identity is asserted, not assumed.

WHAT EACH OUTCOME MEANS.

  - class mix constant and timing decays  -> declining marginal value is wrong
    too, and the mechanism is something this instrument does not see;
  - class mix shifts and timing tracks it -> a mechanism, not a curve;
  - class mix shifts and timing does not  -> byte value is flat across these
    classes, and the constant-slope model survives on reach alone.

THE KV FRACTION IS AN ESTIMATE AND IS LABELLED AS ONE. The probe reports log2
size classes, not exact sizes, so bytes in the `2^24` class are attributed at
the exact observed page size of 17,825,792 B. The unattributed remainder is
reported next to it rather than hidden, because in rung 10 the class lower
bounds summed to 377.6 MiB against a measured 448.0 MiB delta.

Usage
-----
    python3 research/e130_rung11_class_predict.py \
        --artifact research/e130-artifacts/rung11-admission.json \
        --out research/e130-artifacts/rung11-class-prediction.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MIB = 1024.0 * 1024.0

# One page of per-layer KV: 4,352 tokens x 4,096 B per token per layer.
KV_PAGE_BYTES = 17_825_792
KV_CLASS = "2^24"

# Rung 10a, the only measured point on this response.
ANCHOR_PCT = -0.1968
ANCHOR_ADMITTED_MIB = 448.0
SLOPE_PCT_PER_512MIB = ANCHOR_PCT * 512.0 / ANCHOR_ADMITTED_MIB  # -0.2249

RUNGS = ["s64", "s512", "s1024", "s2048"]


def class_bytes_mib(name: str, count: float) -> float:
    """Bytes a class delta holds, at the exact size when one is known."""
    exponent = int(name.removeprefix("2^"))
    size = KV_PAGE_BYTES if name == KV_CLASS else (1 << exponent)
    return count * size / MIB


def step(mass: dict, lo: str, hi: str) -> dict | None:
    """Composition of the bytes that moving from ``lo`` to ``hi`` admits."""
    roles = sorted({key.split("/w")[1] for key in mass})
    per_role = {}
    for role in roles:
        small = mass.get(f"{lo}/w{role}")
        large = mass.get(f"{hi}/w{role}")
        if small is None or large is None:
            continue
        by_class = {}
        for name in sorted(set(small["class_count_mean"])
                           | set(large["class_count_mean"]),
                           key=lambda n: -int(n.removeprefix("2^"))):
            delta = (small["class_count_mean"].get(name, 0.0)
                     - large["class_count_mean"].get(name, 0.0))
            if abs(delta) >= 0.5:
                by_class[name] = {
                    "count_delta": delta,
                    "bytes_mib": class_bytes_mib(name, delta),
                }
        admitted = small["unwired_mib_mean"] - large["unwired_mib_mean"]
        attributed = sum(v["bytes_mib"] for v in by_class.values())
        kv = by_class.get(KV_CLASS, {}).get("bytes_mib", 0.0)
        per_role[f"w{role}"] = {
            "admitted_mib": admitted,
            "admitted_count": (small["unwired_count_mean"]
                               - large["unwired_count_mean"]),
            "by_class": by_class,
            "attributed_mib": attributed,
            "unattributed_mib": admitted - attributed,
            "kv_class_mib": kv,
            "kv_fraction_of_admitted": kv / admitted if admitted else None,
        }
    if not per_role:
        return None

    # The scored worker is w1. Roles are never pooled, but the ladder ships one
    # number, so the scored role leads and the others are reported beside it.
    scored = per_role.get("w1")
    return {
        "from": lo,
        "to": hi,
        "slack_delta_mib": float(int(hi[1:]) - int(lo[1:])),
        "per_role": per_role,
        "scored_role_admitted_mib": scored["admitted_mib"] if scored else None,
        "scored_role_kv_fraction":
            scored["kv_fraction_of_admitted"] if scored else None,
        "admitted_mib_mean_over_roles":
            sum(r["admitted_mib"] for r in per_role.values()) / len(per_role),
        "kv_fraction_mean_over_roles": (
            sum(r["kv_fraction_of_admitted"] for r in per_role.values()
                if r["kv_fraction_of_admitted"] is not None)
            / max(1, sum(1 for r in per_role.values()
                         if r["kv_fraction_of_admitted"] is not None))),
    }


def predict(current: dict, anchor: dict) -> dict:
    """F17 section 3, applied to the mean over roles."""
    admitted = current["admitted_mib_mean_over_roles"]
    kv_now = current["kv_fraction_mean_over_roles"]
    kv_anchor = anchor["kv_fraction_mean_over_roles"]
    ratio = kv_now / kv_anchor if kv_anchor else None
    scaled = SLOPE_PCT_PER_512MIB * admitted / 512.0
    return {
        "admitted_mib": admitted,
        "kv_fraction": kv_now,
        "kv_fraction_at_anchor": kv_anchor,
        "kv_fraction_ratio": ratio,
        "constant_slope_pct": scaled,
        "class_weighted_pct": scaled * ratio if ratio is not None else None,
        "slope_pct_per_512mib": SLOPE_PCT_PER_512MIB,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path,
                    default=Path("research/e130-artifacts/rung11-admission.json"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    mass = json.loads(args.artifact.read_text())[
        "composition_verdict"]["steady_unwired_by_role"]

    steps = {}
    for lo, hi in zip(RUNGS, RUNGS[1:]):
        result = step(mass, lo, hi)
        if result:
            steps[f"{lo}_to_{hi}"] = result
    anchor = steps.get("s64_to_s512")
    if anchor is None:
        raise SystemExit("the s64 to s512 anchor step is missing; cannot scale")

    # The formula must reproduce the rung 10a measurement at the anchor.
    check = predict(anchor, anchor)
    identity_error = abs(check["class_weighted_pct"] - ANCHOR_PCT)

    report = {
        "experiment": "e130-rung11-class-prediction",
        "harness": "local",
        "made_before": "the rung 11 timing legs",
        "kv_page_bytes": KV_PAGE_BYTES,
        "kv_class": KV_CLASS,
        "slope_pct_per_512mib": SLOPE_PCT_PER_512MIB,
        "anchor_measured_pct": ANCHOR_PCT,
        "anchor_identity_error_pct": identity_error,
        "anchor_identity_holds": identity_error < 0.02,
        "steps": steps,
        "predictions": {name: predict(s, anchor) for name, s in steps.items()},
    }

    print("=== e130 rung 11: class-table prediction, made before timing ===")
    print(f"  slope anchor  {SLOPE_PCT_PER_512MIB:+.4f} % per 512 MiB"
          f"  (rung 10a: {ANCHOR_PCT:+.4f} % over {ANCHOR_ADMITTED_MIB:.0f} MiB)")
    print(f"  identity      anchor reproduces {check['class_weighted_pct']:+.4f} %"
          f"  error {identity_error:.4f} pp"
          f"  {'OK' if report['anchor_identity_holds'] else 'FAILED'}")
    for name, s in steps.items():
        p = report["predictions"][name]
        print(f"\n  --- {name}  (slack +{s['slack_delta_mib']:.0f} MiB) ---")
        print(f"    admitted        {p['admitted_mib']:8.1f} MiB mean over roles"
              f"   response {p['admitted_mib'] / s['slack_delta_mib']:.3f} per MiB")
        print(f"    KV class share  {100 * p['kv_fraction']:7.2f} %"
              f"   ratio to anchor {p['kv_fraction_ratio']:.3f}")
        print(f"    PREDICTED       constant slope {p['constant_slope_pct']:+.4f} %"
              f"   class weighted {p['class_weighted_pct']:+.4f} %")
        for role, r in s["per_role"].items():
            classes = " ".join(
                f"{n}{v['count_delta']:+.1f}" for n, v in r["by_class"].items())
            print(f"    {role}  admitted {r['admitted_mib']:8.1f} MiB"
                  f"  KV {100 * (r['kv_fraction_of_admitted'] or 0):5.1f} %"
                  f"  unattributed {r['unattributed_mib']:+7.1f} MiB  [{classes}]")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
