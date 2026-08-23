#!/usr/bin/env python3
"""Re-price F22 against a live receipt's per-prompt raw vector, Rule 121 style.

The median over eight ratios is not a smooth function of those ratios, so a
mechanism's value depends on WHICH two prompts sit in the middle after the
gain is applied. This applies the F22 gain shape to a named receipt's raw
vector, re-sorts, and reports the new median.

The beagle gain is back-solved from the advisor's own F22 point estimate on
`1760479a`, so the shape and the scale both come from F24 section 5 and
nothing here is refitted.
"""
import argparse
import json
import os
import urllib.request

BENCH = "5d1ee4d7-80bd-4555-b182-6505f26ef495"
API = f"https://api.yukon.org/api/benchmarks/{BENCH}/submissions?all=true"

NAMES = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}

# F24 section 5, relative to beagle = 1.000.
GAIN_SHAPE = {
    "beagle": 1.000, "essays": 1.918, "drama": 2.008, "travel": 1.553,
    "botany": 0.952, "medicine": 0.783, "republic": 0.544, "plutarch": 0.002,
}
# Back-solved from the advisor's F22 point estimate: beagle 3.5503 -> 3.56471.
BEAGLE_GAIN_PCT = 0.4059


def fetch():
    request = urllib.request.Request(
        API, headers={"Authorization": "Bearer " + os.environ["YUKON_API_TOKEN"]})
    with urllib.request.urlopen(request) as response:
        body = json.load(response)
    return body.get("submissions", body) if isinstance(body, dict) else body


def raw_vector(row):
    out = {}
    for prompt in row["officialMetrics"]["per_prompt"]:
        key = prompt["prompt_sha256"][:8]
        out[NAMES[key]] = prompt["raw_ratio_of_means"]
    return out


def median_pair(vector):
    ordered = sorted(vector.items(), key=lambda kv: kv[1])
    lower, upper = ordered[3], ordered[4]
    return ordered, lower, upper, (lower[1] + upper[1]) / 2


def report(label, vector):
    ordered, lower, upper, median = median_pair(vector)
    print(f"--- {label} ---")
    for rank, (name, value) in enumerate(ordered):
        mark = "  <== median pair" if rank in (3, 4) else ""
        print(f"  {rank} {name:<9} {value:.5f}{mark}")
    print(f"  median {median:.8f}")
    return ordered, lower, upper, median


parser = argparse.ArgumentParser()
parser.add_argument("receipt", help="receipt id prefix to anchor on")
parser.add_argument("--gain-pct", type=float, default=BEAGLE_GAIN_PCT)
args = parser.parse_args()

row = next(r for r in fetch() if r["id"].startswith(args.receipt))
base = raw_vector(row)
print(f"anchor {row['id'][:8]} {row['solverUsername']} "
      f"official {row['officialScore']}")
_, _, base_upper, base_median = report("anchor", base)

after = {
    name: value * (1 + GAIN_SHAPE[name] * args.gain_pct / 100)
    for name, value in base.items()
}
_, _, after_upper, after_median = report(
    f"after F22 at beagle +{args.gain_pct:.4f} %", after)

print(f"\nmedian delta   {(after_median / base_median - 1) * 100:+.4f} %")
print(f"upper slot     {base_upper[0]} -> {after_upper[0]}")

ordered = sorted(base.items(), key=lambda kv: kv[1])
gap = (ordered[5][1] / ordered[4][1] - 1) * 100
print(f"upper buffer   {ordered[4][0]} sits {gap:.4f} % below {ordered[5][0]}, "
      f"and F22 pays it {GAIN_SHAPE[ordered[4][0]]:.3f} of the beagle gain")
