#!/usr/bin/env python3
"""Emit every pre-registered E135 tripwire for one receipt, digits first.

The advisor pre-registered four tripwires for submission `1db9d63e` and asked
for the digits before any interpretation. Reading them by hand under time
pressure invites a selective readout, so this prints all four from one call,
in a fixed order, whatever they say.

Advisor F26 added one amendment: report the candidate leg against both the
`1760479a` anchor and the `684821ed` bar, and compare against the de-lucked
pack rather than the published medians. De-lucking replaces every serial leg
with the window median, which removes the serial lottery that the candidate
cannot influence.
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

# Pre-registered before the run, recorded here so the readout cannot drift.
FORECAST_MEDIAN = 3.699
FORECAST_BAND = (3.683, 3.716)
MEDPAIR_FORECAST_PCT = 0.12
MEDPAIR_BAND_PCT = (-0.03, 0.27)
PLUTARCH_EDL_EXPECTED = 0.155
PLUTARCH_NONDRAFT_EXPECTED = 449
EXPECTED_PAIR = ("beagle", "essays")


def fetch():
    request = urllib.request.Request(
        API, headers={"Authorization": "Bearer " + os.environ["YUKON_API_TOKEN"]})
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.load(response)
    return body["submissions"]


def per_prompt(row):
    out = {}
    for prompt in row["officialMetrics"]["per_prompt"]:
        out[NAMES[prompt["prompt_sha256"][:8]]] = prompt
    return out


def complete(row):
    metrics = row.get("officialMetrics") or {}
    prompts = metrics.get("per_prompt") or []
    if len(prompts) != len(NAMES):
        return False
    return all(p.get("serial_seconds_per_token_mean") for p in prompts)


def pick(rows, prefix):
    hits = [r for r in rows if r["id"].startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"{prefix} matched {len(hits)} rows")
    return hits[0]


def sorted_raw(prompts):
    return sorted(
        ((name, p["raw_ratio_of_means"]) for name, p in prompts.items()),
        key=lambda kv: kv[1])


def median_of(ordered):
    return (ordered[3][1] + ordered[4][1]) / 2


def de_lucked(prompts, serial_median):
    """Median with every serial leg replaced by the window median."""
    ratios = sorted(
        serial_median[name] / p["mtp_seconds_per_token_mean"]
        for name, p in prompts.items())
    return (ratios[3] + ratios[4]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("receipt", help="receipt id prefix to read out")
    ap.add_argument("--anchor", default="1760479a")
    ap.add_argument("--bar", default="684821ed")
    ap.add_argument("--pool-since", default="2026-08-22T18:00:00Z")
    args = ap.parse_args()

    rows = fetch()
    row = pick(rows, args.receipt)

    print("=" * 66)
    print(f"RECEIPT {row['id']}")
    print(f"  status            {row.get('status')}")
    print(f"  promotionStatus   {row.get('promotionStatus')}")
    print(f"  officialScore     {row.get('officialScore')}")
    print(f"  commit            {row.get('submissionCommitSha')}")
    print(f"  created           {row.get('createdAt')}")
    if row.get("rejectionReason"):
        print(f"  rejectionReason   {row['rejectionReason']}")
    print("=" * 66)

    if not complete(row):
        raise SystemExit("\nno complete per-prompt metrics on this row yet")

    prompts = per_prompt(row)
    anchor = pick(rows, args.anchor)
    bar = pick(rows, args.bar)
    anchor_p, bar_p = per_prompt(anchor), per_prompt(bar)

    ordered = sorted_raw(prompts)
    median = median_of(ordered)

    print("\n-- full sorted raw vector, ours --")
    for rank, (name, value) in enumerate(ordered):
        mark = "  <== median pair" if rank in (3, 4) else ""
        print(f"  {rank} {name:<9} {value:.6f}{mark}")
    print(f"  median {median:.8f}")

    print("\n-- TRIPWIRE 1: plutarch schedule, did the revert reach the binary --")
    plutarch = prompts["plutarch"]
    edl = plutarch["effective_mean_draft_len"]
    nondraft = plutarch["non_drafting_round_count"]
    print(f"  effective_mean_draft_len   {edl}")
    print(f"  non_drafting_round_count   {nondraft}")
    print(f"  expected                   edl about {PLUTARCH_EDL_EXPECTED}, "
          f"non-drafting about {PLUTARCH_NONDRAFT_EXPECTED}")
    void = nondraft == 0 and edl > 1.0
    print(f"  VERDICT                    "
          f"{'ROW IS VOID, the revert did not reach the binary' if void else 'revert reached the binary'}")

    print("\n-- TRIPWIRE 2: median pair identity --")
    got = (ordered[3][0], ordered[4][0])
    print(f"  realised pair              {got[0]} then {got[1]}")
    print(f"  pre-registered             {EXPECTED_PAIR[0]} then {EXPECTED_PAIR[1]}")
    print(f"  VERDICT                    "
          f"{'as pre-registered' if got == EXPECTED_PAIR else 'DIFFERS from pre-registration'}")

    print("\n-- TRIPWIRE 3: candidate leg, the headline --")
    print("  candidate mtp seconds per token, lower is better")
    print(f"  {'prompt':<9} {'ours':>12} {args.anchor:>12} {'vs anchor':>10}"
          f" {args.bar:>12} {'vs bar':>10}")
    for name in EXPECTED_PAIR:
        ours = prompts[name]["mtp_seconds_per_token_mean"]
        a = anchor_p[name]["mtp_seconds_per_token_mean"]
        b = bar_p[name]["mtp_seconds_per_token_mean"]
        print(f"  {name:<9} {ours:12.7f} {a:12.7f} {(ours/a-1)*100:+9.3f}%"
              f" {b:12.7f} {(ours/b-1)*100:+9.3f}%")
    for label, other in ((args.anchor, anchor_p), (args.bar, bar_p)):
        ours_mean = statistics.mean(
            prompts[n]["mtp_seconds_per_token_mean"] for n in EXPECTED_PAIR)
        other_mean = statistics.mean(
            other[n]["mtp_seconds_per_token_mean"] for n in EXPECTED_PAIR)
        delta = (ours_mean / other_mean - 1) * 100
        inside = MEDPAIR_BAND_PCT[0] <= delta <= MEDPAIR_BAND_PCT[1]
        print(f"  MEDPAIR MEAN vs {label}   {delta:+.3f} %"
              f"   forecast {MEDPAIR_FORECAST_PCT:+.2f} %"
              f"   band [{MEDPAIR_BAND_PCT[0]:+.2f}, {MEDPAIR_BAND_PCT[1]:+.2f}]"
              f"   {'INSIDE' if inside else 'OUTSIDE'}")

    print("\n  serial leg beside it, noise diagnostic only")
    for label, other in ((args.anchor, anchor_p), (args.bar, bar_p)):
        ours_mean = statistics.mean(
            prompts[n]["serial_seconds_per_token_mean"] for n in EXPECTED_PAIR)
        other_mean = statistics.mean(
            other[n]["serial_seconds_per_token_mean"] for n in EXPECTED_PAIR)
        print(f"  serial medpair vs {label} {(ours_mean/other_mean-1)*100:+.3f} %")

    print("\n-- TRIPWIRE 4: published median --")
    inside = FORECAST_BAND[0] <= median <= FORECAST_BAND[1]
    print(f"  published                  {median:.8f}")
    print(f"  forecast                   {FORECAST_MEDIAN}"
          f"   band [{FORECAST_BAND[0]}, {FORECAST_BAND[1]}]")
    print(f"  VERDICT                    {'INSIDE' if inside else 'OUTSIDE'}")
    print(f"  live bar {args.bar}      {bar['officialScore']}")
    print(f"  gap to bar                 {(median/float(bar['officialScore'])-1)*100:+.3f} %")

    print("\n-- de-lucked comparison, F26 amendment --")
    pool = [r for r in rows
            if complete(r) and r["createdAt"] >= args.pool_since]
    serial_median = {
        name: statistics.median(
            per_prompt(r)[name]["serial_seconds_per_token_mean"] for r in pool)
        for name in NAMES.values()}
    print(f"  window {len(pool)} rows since {args.pool_since}")
    for label, target in (("ours", prompts), (args.anchor, anchor_p),
                          (args.bar, bar_p)):
        print(f"  {label:<10} de-lucked {de_lucked(target, serial_median):.6f}")
    print("  de-lucking removes the serial draw, which the candidate cannot "
          "influence")


main()
