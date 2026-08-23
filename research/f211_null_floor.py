#!/usr/bin/env python3
"""FINDING 211 -- the published-median noise floor is the SERIAL leg.

The draft schedule is deterministic given prompt and head, so two rows sharing
a draft-length vector prove nothing about the candidate code.  A true null pair
must instead be found by tightness: if the eight per-prompt candidate times
agree with a residual scatter far below any known mechanism, the two candidate
binaries are doing the same work.

For every such pair, report how much of the published-median move came from
the candidate leg and how much came from the serial leg the candidate cannot
touch.
"""
import json
import math
import statistics
import sys

BOARD = "/tmp/yukon-board/full.json"
SINCE = sys.argv[1] if len(sys.argv) > 1 else "2026-08-22T15:00:00Z"
TIGHT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15

PROMPTS = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}
MEDPAIR = ("beagle", "essays")

with open(BOARD) as fh:
    rows = json.load(fh)["submissions"]


def vec(row):
    pp = (row.get("officialMetrics") or {}).get("per_prompt")
    if not pp or len(pp) != 8:
        return None
    out = {}
    for e in pp:
        n = PROMPTS.get(e["prompt_sha256"][:8])
        if n is None or e.get("mtp_seconds_per_token_mean") is None:
            return None
        out[n] = e
    return out if len(out) == 8 else None


scored = []
for r in rows:
    v = vec(r)
    if v is None or (r.get("createdAt") or "") < SINCE:
        continue
    if v["beagle"]["effective_mean_draft_len"] < 3.0:
        continue
    scored.append((r, v))

print(f"window since {SINCE}, rows {len(scored)}, tightness cut {TIGHT} %")
print()

hdr = (
    f"{'A':<10}{'B':<10}{'solverA':<15}{'solverB':<15}"
    f"{'candmean%':>10}{'candsd%':>9}{'sermean%':>10}{'sersd%':>8}"
    f"{'pub d%':>9}{'pred d%':>9}"
)
print(hdr)
print("-" * len(hdr))

cand_means, ser_means, pub_ds, resid = [], [], [], []
for i in range(len(scored)):
    for j in range(i + 1, len(scored)):
        ra, va = scored[i]
        rb, vb = scored[j]
        cd = [
            100.0
            * (
                vb[n]["mtp_seconds_per_token_mean"]
                / va[n]["mtp_seconds_per_token_mean"]
                - 1.0
            )
            for n in va
        ]
        cs = statistics.stdev(cd)
        if cs > TIGHT:
            continue
        sd_ = [
            100.0
            * (
                vb[n]["serial_seconds_per_token_mean"]
                / va[n]["serial_seconds_per_token_mean"]
                - 1.0
            )
            for n in va
        ]
        cm, sm, ss = statistics.fmean(cd), statistics.fmean(sd_), statistics.stdev(sd_)
        sa, sb = ra.get("officialScore"), rb.get("officialScore")
        if not (sa and sb):
            continue
        pub = 100.0 * (sb / sa - 1.0)
        # medpair prediction: published delta ~= serial medpair - candidate medpair
        mp_c = statistics.fmean(
            [
                100.0
                * (
                    vb[n]["mtp_seconds_per_token_mean"]
                    / va[n]["mtp_seconds_per_token_mean"]
                    - 1.0
                )
                for n in MEDPAIR
            ]
        )
        mp_s = statistics.fmean(
            [
                100.0
                * (
                    vb[n]["serial_seconds_per_token_mean"]
                    / va[n]["serial_seconds_per_token_mean"]
                    - 1.0
                )
                for n in MEDPAIR
            ]
        )
        pred = mp_s - mp_c
        cand_means.append(cm)
        ser_means.append(sm)
        pub_ds.append(pub)
        resid.append(pub - pred)
        print(
            f"{ra['id'][:8]:<10}{rb['id'][:8]:<10}"
            f"{(ra.get('solverUsername') or '?')[:14]:<15}"
            f"{(rb.get('solverUsername') or '?')[:14]:<15}"
            f"{cm:>10.4f}{cs:>9.4f}{sm:>10.4f}{ss:>8.4f}{pub:>9.4f}{pred:>9.4f}"
        )

n = len(cand_means)
print()
print(f"true near-null candidate pairs                {n}")
if n > 1:
    print(
        f"candidate 8-prompt mean   spread  sd        "
        f"{statistics.stdev(cand_means):>8.4f} %"
    )
    print(
        f"serial    8-prompt mean   spread  sd        "
        f"{statistics.stdev(ser_means):>8.4f} %"
    )
    print(
        f"published median delta    spread  sd        "
        f"{statistics.stdev(pub_ds):>8.4f} %"
    )
    print(
        f"residual after serial-minus-candidate model  "
        f"{statistics.stdev(resid):>8.4f} %"
    )
    print()
    print(
        "ratio of published-median scatter to candidate scatter  "
        f"{statistics.stdev(pub_ds)/statistics.stdev(cand_means):>6.2f} x"
    )
