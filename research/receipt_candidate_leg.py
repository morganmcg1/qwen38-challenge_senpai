#!/usr/bin/env python3
"""Price a ranked contrast on the candidate leg, per RULE 177.

The published median is an order statistic over four near-tied prompts and it
carries the serial-leg lottery, which no candidate edit can touch (FINDING 305,
FINDING 306). Price mechanisms on `mtp_seconds_per_token_mean`, paired across
the eight prompts.

    export YUKON_API_TOKEN=...
    python3 research/receipt_candidate_leg.py fetch
    python3 research/receipt_candidate_leg.py compare <id_a> <id_b>
    python3 research/receipt_candidate_leg.py serial <id>
"""
import json
import os
import statistics as st
import sys
import urllib.request

BENCHMARK = "5d1ee4d7-80bd-4555-b182-6505f26ef495"
CACHE = os.environ.get("RECEIPT_CACHE", "/tmp/yukon_board.json")
NAME = {
    "c1ec5866": "plutarch", "4b9e88cd": "drama", "3b10cb4d": "travel",
    "919318e1": "beagle", "00142a44": "medicine", "ea82dcb5": "republic",
    "a2ea8b60": "essays", "192fb621": "botany",
}


def fetch():
    token = os.environ["YUKON_API_TOKEN"]
    url = (f"https://api.yukon.org/api/benchmarks/{BENCHMARK}/submissions"
           "?all=true")
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    with open(CACHE, "w") as handle:
        json.dump(payload, handle)
    return payload


def load():
    if not os.path.exists(CACHE):
        return fetch()
    with open(CACHE) as handle:
        return json.load(handle)


def submissions():
    payload = load()
    rows = payload["submissions"] if isinstance(payload, dict) else payload
    return {row["id"][:8]: row for row in rows}


def prompts(row):
    per_prompt = row["officialMetrics"]["per_prompt"]
    return {p["prompt_sha256"][:8]: p for p in per_prompt}


def median8(values):
    values = sorted(values)
    return (values[3] + values[4]) / 2


def compare(id_a, id_b):
    """Paired per-prompt candidate-leg contrast. Negative means B is faster."""
    table = submissions()
    a, b = prompts(table[id_a]), prompts(table[id_b])
    shared = sorted(set(a) & set(b), key=lambda k: a[k]["mtp_seconds_per_token_mean"])
    print(f"{id_a} -> {id_b}   negative = B decoded faster   harness=ranked")
    print("%-9s %11s %11s %9s %9s %9s %6s" % (
        "prompt", "candA", "candB", "cand d%", "ser d%", "raw d%", "edl=="))
    deltas = []
    for key in shared:
        x, y = a[key], b[key]
        cand = (y["mtp_seconds_per_token_mean"] / x["mtp_seconds_per_token_mean"] - 1) * 100
        serial = (y["serial_seconds_per_token_mean"] / x["serial_seconds_per_token_mean"] - 1) * 100
        raw = (y["raw_ratio_of_means"] / x["raw_ratio_of_means"] - 1) * 100
        same = x["effective_mean_draft_len"] == y["effective_mean_draft_len"]
        deltas.append(cand)
        print("%-9s %11.6f %11.6f %+9.4f %+9.4f %+9.4f %6s" % (
            NAME.get(key, key), x["mtp_seconds_per_token_mean"],
            y["mtp_seconds_per_token_mean"], cand, serial, raw, same))
    mean = st.mean(deltas)
    sd = st.stdev(deltas)
    se = sd / len(deltas) ** 0.5
    print()
    print("candidate leg  mean %+.4f %%  sd %.4f  se %.4f  2sigma %.4f  same sign %d/%d" % (
        mean, sd, se, 2 * se, sum(1 for d in deltas if d * mean > 0), len(deltas)))
    # RULE 176: published percent for a uniform saving g is g/(1-g).
    g = -mean / 100
    print("RULE 176 published estimate for a uniform saving: %+.4f %%" % (100 * g / (1 - g)))
    for tag, row in ((id_a, table[id_a]), (id_b, table[id_b])):
        print("%s published median %s" % (tag, row.get("officialScore")))


def serial(sub_id):
    """Flag the serial-leg anomaly that FINDING 305 found in the crown run."""
    table = submissions()
    per_prompt = prompts(table[sub_id])
    values = {NAME.get(k, k): v["serial_seconds_per_token_mean"] for k, v in per_prompt.items()}
    median = st.median(values.values())
    print(f"{sub_id}  serial leg, run median {median:.6f}")
    for name, value in sorted(values.items(), key=lambda kv: -kv[1]):
        flag = "  <-- ANOMALY, inflates this prompt's ratio" if value / median - 1 > 0.005 else ""
        print("  %-9s %.6f  %+.3f %%%s" % (name, value, (value / median - 1) * 100, flag))
    print("within-run spread %.3f %%  (board median 0.533 %%, p90 0.794 %%, p99 1.616 %%)" % (
        (max(values.values()) / min(values.values()) - 1) * 100))


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    if command == "fetch":
        print("cached %d submissions to %s" % (len(fetch()["submissions"]), CACHE))
    elif command == "compare":
        compare(sys.argv[2][:8], sys.argv[3][:8])
    elif command == "serial":
        serial(sys.argv[2][:8])
    else:
        print(__doc__)
        sys.exit(2)
