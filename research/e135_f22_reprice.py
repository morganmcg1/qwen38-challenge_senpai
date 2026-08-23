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
import statistics
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


def leg_vectors(row):
    """Return the per-prompt serial and candidate legs, keyed by prompt name."""
    serial, mtp = {}, {}
    for prompt in row["officialMetrics"]["per_prompt"]:
        name = NAMES[prompt["prompt_sha256"][:8]]
        serial[name] = prompt["serial_seconds_per_token_mean"]
        mtp[name] = prompt["mtp_seconds_per_token_mean"]
    return serial, mtp


def complete(row):
    metrics = row.get("officialMetrics") or {}
    per_prompt = metrics.get("per_prompt") or []
    if len(per_prompt) != len(NAMES):
        return False
    return all(p.get("serial_seconds_per_token_mean") for p in per_prompt)


def draw_gain(serial, anchor_mtp, gain_pct):
    """Apply the shape to one serial draw and return its median gain and slot.

    The candidate leg is held fixed. Only the serial vector varies, because the
    serial draw is the part of a ranked row the candidate cannot influence.
    """
    base = {name: serial[name] / anchor_mtp[name] for name in anchor_mtp}
    after = {
        name: value * (1 + GAIN_SHAPE[name] * gain_pct / 100)
        for name, value in base.items()
    }
    _, _, base_upper, base_median = median_pair(base)
    _, _, _, after_median = median_pair(after)
    return (after_median / base_median - 1) * 100, base_upper[0], base


def crossing_gain(base):
    """Beagle gain at which the shape lifts the rank-4 prompt past rank 5."""
    ordered = sorted(base.items(), key=lambda kv: kv[1])
    (n4, r4), (n5, r5) = ordered[4], ordered[5]
    denominator = r4 * GAIN_SHAPE[n4] - r5 * GAIN_SHAPE[n5]
    if denominator <= 0:
        return n4, n5, None
    return n4, n5, 100 * (r5 - r4) / denominator


def expectation_mode(rows, args):
    anchor = next(r for r in rows if r["id"].startswith(args.receipt))
    _, anchor_mtp = leg_vectors(anchor)
    pool = sorted(
        (r for r in rows if complete(r) and r["createdAt"] >= args.pool_since),
        key=lambda r: r["createdAt"])
    if not pool:
        raise SystemExit("no complete rows in the pool window")

    print(f"anchor candidate leg  {anchor['id'][:8]} {anchor['solverUsername']}")
    print(f"serial pool           {len(pool)} rows since {args.pool_since}")
    print(f"shape                 F22, beagle +{args.gain_pct:.4f} %\n")

    draws, census, crossings = [], {}, {}
    for row in pool:
        serial, _ = leg_vectors(row)
        gain, owner, base = draw_gain(serial, anchor_mtp, args.gain_pct)
        draws.append((row["id"][:8], gain, owner))
        census[owner] = census.get(owner, 0) + 1
        _, _, cross = crossing_gain(base)
        crossings.setdefault(owner, []).append(cross)

    values = [g for _, g, _ in draws]
    print("== F22 expected value over the serial lottery ==")
    print(f"  expected      {statistics.mean(values):+.4f} %")
    print(f"  sd            {statistics.stdev(values):.4f}")
    print(f"  p05           {percentile(values, 5):+.4f} %")
    print(f"  p95           {percentile(values, 95):+.4f} %")
    print(f"  min / max     {min(values):+.4f} % / {max(values):+.4f} %")

    print("\n== upper-slot occupancy, anchor candidate held fixed ==")
    for owner, count in sorted(census.items(), key=lambda kv: -kv[1]):
        print(f"  {owner:<9} {count:3d}   {100 * count / len(pool):5.1f} %")

    print("\n== saturation: beagle gain that changes the median pair ==")
    for owner in sorted(crossings, key=lambda o: -census[o]):
        found = [c for c in crossings[owner] if c is not None]
        share = census[owner] / len(pool)
        if not found:
            print(f"  conditional on {owner:<9} never crosses, uncapped")
            continue
        print(f"  conditional on {owner:<9} {statistics.mean(found):+.3f} % "
              f"beagle gain   (p={share:.3f})")
    weighted = sum(
        statistics.mean([c for c in crossings[o] if c is not None])
        * census[o] / len(pool)
        for o in crossings if any(c is not None for c in crossings[o]))
    print(f"  probability-weighted            {weighted:+.3f} % beagle gain")
    print(f"  shipped F22 gain is +{args.gain_pct:.4f} %, so the shape is "
          f"{'UNCAPPED' if weighted > args.gain_pct else 'CAPPED'}")

    if args.show_draws:
        print("\n== per-draw detail ==")
        for rid, gain, owner in draws:
            print(f"  {rid}  {gain:+.4f} %   upper {owner}")


def percentile(values, pct):
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def single_mode(rows, args):
    row = next(r for r in rows if r["id"].startswith(args.receipt))
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
    print(f"upper buffer   {ordered[4][0]} sits {gap:.4f} % below "
          f"{ordered[5][0]}, and F22 pays it "
          f"{GAIN_SHAPE[ordered[4][0]]:.3f} of the beagle gain")
    print("\nRule 126: this is one draw of the serial lottery, not a price. "
          "Use --expectation to price the mechanism.")


parser = argparse.ArgumentParser()
parser.add_argument("receipt", help="receipt id prefix to anchor on")
parser.add_argument("--gain-pct", type=float, default=BEAGLE_GAIN_PCT)
parser.add_argument("--expectation", action="store_true",
                    help="price over the serial lottery, per Rule 126")
parser.add_argument("--pool-since", default="2026-08-22T18:00:00Z",
                    help="earliest createdAt for a pooled serial vector")
parser.add_argument("--show-draws", action="store_true")
args = parser.parse_args()

rows = fetch()
if args.expectation:
    expectation_mode(rows, args)
else:
    single_mode(rows, args)
