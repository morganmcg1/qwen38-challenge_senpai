#!/usr/bin/env python3
"""Split our remaining candidate-leg gap into a schedule part and a kernel part.

We are behind the promoted anchor on the candidate leg. That gap has two very
different causes and they need different work, so guessing between them wastes
ranked receipts.

F218 measured a ranked exchange rate near -9.42 % candidate seconds per token
for each +1.0 of realised draft length. A rival that drafts deeper than us
therefore buys candidate time we could buy the same way, and that part of the
gap is a schedule question. Whatever the exchange rate does not explain is a
kernel and runtime question.

The split matters because the two are not equally testable here. Draft length
is bit reproducible on the ranked runner, so the schedule part is measured
without noise, but E137 showed this host inverts the sign of depth price
against the ranked runner, so schedule changes can only be settled on ranked
receipts. Kernel changes keep their sign locally.
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

EDL_EXCHANGE_PCT = -9.42
# F83 ranked prompt weights, the chance each prompt drives the published median.
WEIGHT = {
    "beagle": 0.4862, "medicine": 0.2508, "essays": 0.1598,
    "botany": 0.0124, "republic": 0.0100,
    "plutarch": 0.0, "drama": 0.0, "travel": 0.0,
}


def fetch():
    request = urllib.request.Request(
        API, headers={"Authorization": "Bearer " + os.environ["YUKON_API_TOKEN"]})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)["submissions"]


def per_prompt(row):
    return {NAMES[p["prompt_sha256"][:8]]: p
            for p in row["officialMetrics"]["per_prompt"]}


def pick(rows, prefix):
    hits = [r for r in rows if r["id"].startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"{prefix} matched {len(hits)} rows")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="572b2cc4")
    ap.add_argument("--rival", default="1760479a")
    args = ap.parse_args()

    rows = fetch()
    ours = per_prompt(pick(rows, args.ours))
    rival = per_prompt(pick(rows, args.rival))

    print(f"candidate leg, {args.ours} against {args.rival}")
    print(f"positive gap means we are slower. schedule part is priced at "
          f"{EDL_EXCHANGE_PCT:+.2f} % per +1.0 draft length.")
    print()
    print("%-9s %8s %8s %8s %9s %9s %9s %7s" % (
        "prompt", "our edl", "their", "d edl", "gap %", "schedule", "kernel",
        "weight"))

    rowsout = []
    for name in NAMES.values():
        our_t = ours[name]["mtp_seconds_per_token_mean"]
        their_t = rival[name]["mtp_seconds_per_token_mean"]
        gap = (our_t / their_t - 1.0) * 100.0
        our_d = ours[name]["effective_mean_draft_len"]
        their_d = rival[name]["effective_mean_draft_len"]
        # Their extra depth predicts time they save and we do not.
        schedule = -(their_d - our_d) * EDL_EXCHANGE_PCT
        kernel = gap - schedule
        rowsout.append((name, gap, schedule, kernel))
        print("%-9s %8.4f %8.4f %+8.4f %+9.3f %+9.3f %+9.3f %7.4f" % (
            name, our_d, their_d, our_d - their_d, gap, schedule, kernel,
            WEIGHT[name]))

    print()
    for label, index in (("gap", 1), ("schedule", 2), ("kernel", 3)):
        eight = statistics.fmean(r[index] for r in rowsout)
        weighted = sum(r[index] * WEIGHT[r[0]] for r in rowsout)
        print("%-9s   eight prompt mean %+7.3f %%   F83 weighted %+7.3f %%"
              % (label, eight, weighted))

    print()
    print("F83 weighted uses the chance each prompt drives the published "
          "median, so it prices the gap the way the score sees it.")


main()
