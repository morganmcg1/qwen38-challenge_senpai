#!/usr/bin/env python3
"""E184 six-arm ABBA session reduction.

Answers two questions that need one thermal envelope:
  1. Does removing the Seatbelt profile move prefill timing? (advisor F8 s2)
  2. What does each profiler granularity cost end to end?

The `off` arms carry no phase records by construction, so they are the
uncontaminated anchor for pricing every mechanism (advisor F2 s4).
"""

import glob
import json
import statistics as st

SANDBOXED_ANCHOR = 3.9123110
MODES = {
    "off": ["a1-off", "a6-off"],
    "coarse": ["a2-coarse", "a5-coarse"],
    "fine": ["a3-fine", "a4-fine"],
}


def seed_seconds(arms):
    values = []
    for arm in arms:
        for path in sorted(glob.glob(f"research/capture-e184-{arm}/*.json")):
            try:
                doc = json.load(open(path))
            except Exception:
                continue
            if "seed_prefill_seconds" in doc:
                values.append(doc["seed_prefill_seconds"])
    return values


def describe(values):
    mean = st.mean(values)
    sd = st.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "sd": sd,
        "rel_sd_pct": 100.0 * sd / mean,
        "min": min(values),
        "max": max(values),
    }


def main():
    stats = {name: describe(seed_seconds(arms)) for name, arms in MODES.items()}
    off = stats["off"]

    sandbox = {
        "unsandboxed_off_mean_seconds": off["mean"],
        "sandboxed_anchor_seconds": SANDBOXED_ANCHOR,
        "delta_seconds": off["mean"] - SANDBOXED_ANCHOR,
        "delta_pct": 100.0 * (off["mean"] - SANDBOXED_ANCHOR) / SANDBOXED_ANCHOR,
        "off_arm_rel_sd_pct": off["rel_sd_pct"],
        "inside_noise": abs(off["mean"] - SANDBOXED_ANCHOR) / SANDBOXED_ANCHOR
        < off["rel_sd_pct"] / 100.0,
    }

    overhead = {
        mode: 100.0 * (stats[mode]["mean"] / off["mean"] - 1.0)
        for mode in ("coarse", "fine")
    }

    drift = {}
    for mode, arms in MODES.items():
        first, last = st.mean(seed_seconds([arms[0]])), st.mean(seed_seconds([arms[1]]))
        drift[mode] = {
            "first_arm": arms[0], "first_mean": first,
            "last_arm": arms[1], "last_mean": last,
            "drift_pct": 100.0 * (last / first - 1.0),
        }

    report = {
        "harness": "local",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "seed_prefill_seconds_by_mode": stats,
        "sandbox_comparison": sandbox,
        "instrument_overhead_pct": overhead,
        "abba_drift": drift,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    with open("research/e184-session-reduction.json", "w") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
