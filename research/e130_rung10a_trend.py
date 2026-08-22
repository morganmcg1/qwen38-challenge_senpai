#!/usr/bin/env python3
"""Re-analyse rung 10a under arm means plus one linear session trend (F14).

The palindrome ``none s64 s512 s512 s64 none`` makes arm identity collinear
with distance from the session centre, so a within-arm spread is a drift
readout and not a dispersion instrument (campaign Rule 102). The same symmetry
makes the trend exactly orthogonal to every arm contrast, so the arm means are
unbiased and the residual is a clean 2-df error term.

Fits candidate time, the co-timed serial control, and the local ratio, so the
serial leg can be read on the treatment contrast and not only on the control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARTIFACT = Path("research/e130-artifacts/rung10a-wiring-ladder.json")
ARMS = ("none", "s64", "s512")
CENTRE = 3.5
RESIDUAL_DF = 2
T_CRIT_95_2DF = 4.302653


def fit(positions: list[float], values: list[float], arms: list[str]) -> dict:
    """Least squares for ``y = mean(arm) + slope * (position - centre)``.

    The palindrome makes the trend column orthogonal to all three arm
    indicators, so the normal equations separate and no matrix solve is needed.
    """
    offsets = [p - CENTRE for p in positions]
    for arm in ARMS:
        lever = sum(o for o, a in zip(offsets, arms) if a == arm)
        if abs(lever) > 1e-12:
            raise SystemExit(f"design is not orthogonal for arm {arm}: {lever}")

    means = {
        arm: sum(v for v, a in zip(values, arms) if a == arm)
        / sum(1 for a in arms if a == arm)
        for arm in ARMS
    }
    centred = [v - means[a] for v, a in zip(values, arms)]
    slope = sum(o * c for o, c in zip(offsets, centred)) / sum(o * o for o in offsets)
    residuals = [c - slope * o for c, o in zip(centred, offsets)]

    sse = sum(r * r for r in residuals)
    sigma = (sse / RESIDUAL_DF) ** 0.5
    grand = sum(values) / len(values)

    # Fraction of the arm-means-only residual that the single slope explains.
    ss_arm_only = sum(c * c for c in centred)
    explained = 1.0 - sse / ss_arm_only if ss_arm_only > 0 else float("nan")

    return {
        "arm_means": means,
        "slope_per_leg": slope,
        "slope_pct_per_leg": 100.0 * slope / grand,
        "residuals": residuals,
        "residual_sd": sigma,
        "residual_sd_pct": 100.0 * sigma / grand,
        # Both arm means average two legs, so se(difference) = sigma * sqrt(1/2+1/2).
        "se_difference": sigma,
        "se_difference_pct": 100.0 * sigma / grand,
        "trend_share_of_model_a_residual": explained,
        "grand_mean": grand,
    }


def contrast(model: dict, lo: str, hi: str) -> dict:
    """Signed effect of moving from ``lo`` to ``hi``, in percent of ``lo``."""
    base = model["arm_means"][lo]
    delta = model["arm_means"][hi] - base
    pct = 100.0 * delta / base
    se_pct = 100.0 * model["se_difference"] / base
    half = T_CRIT_95_2DF * se_pct
    return {
        "from": lo,
        "to": hi,
        "delta": delta,
        "pct": pct,
        "t": pct / se_pct,
        "ci95_pct": [pct - half, pct + half],
        "significant_at_2df": abs(pct / se_pct) > T_CRIT_95_2DF,
    }


def spread_table(positions: list[float], values: list[float], arms: list[str],
                 slope: float) -> list[dict]:
    """Within-arm spread against what the single session slope predicts.

    A ratio near 1.0 means the spread is drift, not arm behaviour.
    """
    rows = []
    for arm in ARMS:
        pts = [(p, v) for p, v, a in zip(positions, values, arms) if a == arm]
        pts.sort()
        separation = pts[-1][0] - pts[0][0]
        observed = abs(pts[-1][1] - pts[0][1])
        predicted = abs(slope * separation)
        rows.append({
            "arm": arm,
            "separation_legs": separation,
            "observed": observed,
            "trend_predicts": predicted,
            "ratio": observed / predicted if predicted else float("nan"),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    legs = json.loads(args.artifact.read_text())["legs"]
    legs.sort(key=lambda leg: leg["index"])
    positions = [float(leg["index"]) for leg in legs]
    arms = [leg["arm"] for leg in legs]

    channels = {
        "candidate_mtp_seconds_per_token": "mtp_seconds_per_token",
        "serial_seconds_per_token": "serial_seconds_per_token",
        "local_ratio_mtp_decode_speedup": "mtp_decode_speedup",
    }

    report = {
        "experiment": "e130-rung10a-trend",
        "model": "arm means + one linear session trend, 2 residual df",
        "rule": "Rule 102: a palindrome cannot measure within-arm dispersion",
        "harness": "local",
        "order": arms,
        "channels": {},
    }

    for name, field in channels.items():
        values = [float(leg[field]) for leg in legs]
        model = fit(positions, values, arms)
        report["channels"][name] = {
            **{k: v for k, v in model.items() if k != "residuals"},
            "residuals": dict(zip((leg["tag"] for leg in legs), model["residuals"])),
            "within_arm_spread": spread_table(positions, values, arms,
                                              model["slope_per_leg"]),
            "contrasts": {
                "control_none_to_s512": contrast(model, "none", "s512"),
                "none_to_s64": contrast(model, "none", "s64"),
                "treatment_s64_to_s512": contrast(model, "s64", "s512"),
            },
        }

    for name, block in report["channels"].items():
        print(f"\n=== {name} ===")
        print(f"  slope   {block['slope_per_leg']:+.6e} per leg"
              f"  = {block['slope_pct_per_leg']:+.4f} % per leg")
        print(f"  resid sd {block['residual_sd']:.6e}"
              f"  = {block['residual_sd_pct']:.4f} %"
              f"  (trend explains {100 * block['trend_share_of_model_a_residual']:.2f} %"
              " of the arm-means-only residual)")
        for arm in ARMS:
            print(f"  mean {arm:>5}  {block['arm_means'][arm]:.8f}")
        for row in block["within_arm_spread"]:
            print(f"  spread {row['arm']:>5}  sep {row['separation_legs']:.0f}"
                  f"  obs {row['observed']:.4e}"
                  f"  trend {row['trend_predicts']:.4e}"
                  f"  ratio {row['ratio']:.3f}")
        for key, c in block["contrasts"].items():
            print(f"  {key:<24} {c['pct']:+.4f} %  t={c['t']:+.2f}"
                  f"  95% CI [{c['ci95_pct'][0]:+.4f}, {c['ci95_pct'][1]:+.4f}]"
                  f"  {'SIG' if c['significant_at_2df'] else 'ns'}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
