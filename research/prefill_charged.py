#!/usr/bin/env python3
"""Is prefill charged to raw_ratio_of_means?

If raw_ratio_of_means == serial_seconds_per_token_mean / mtp_seconds_per_token_mean
exactly, then prefill_seconds_per_token is reported but NOT scored, and every
prefill optimisation is worth exactly zero to the score.
"""
import json
import os
import urllib.request

NAMES = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}


def get(sub_id):
    tok = os.environ.get("YUKON_API_TOKEN", "")
    url = "https://api.yukon.org/api/submissions/" + sub_id
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read().decode())
    return d.get("submission", d)


live = json.load(open("/tmp/rows_live.json"))
scored = [r for r in live if isinstance(r.get("officialScore"), (int, float))]
scored.sort(key=lambda r: -r["officialScore"])

ours = json.load(open("/tmp/ca92.json"))
ours = ours.get("submission", ours)

targets = [("OURS ca9251b8", ours)]
for r in scored[:3]:
    targets.append(("rank%d %s" % (scored.index(r) + 1, r["id"][:8]), get(r["id"])))

print("Hypothesis A: raw = serial/mtp                (prefill EXCLUDED)")
print("Hypothesis B: raw = (pf+serial)/(pf+mtp)      (prefill INCLUDED)")
print()
hdr = "%-16s %-9s %14s %14s %14s %10s" % (
    "row", "prompt", "raw_reported", "A serial/mtp", "B with prefill", "A relerr")
print(hdr)
print("-" * len(hdr))
worstA = 0.0
nA = 0
for label, sub in targets:
    for p in sorted(sub["officialMetrics"]["per_prompt"],
                    key=lambda p: p["effective_mean_draft_len"]):
        nm = NAMES[p["prompt_sha256"][:8]]
        raw = p["raw_ratio_of_means"]
        s = p["serial_seconds_per_token_mean"]
        m = p["mtp_seconds_per_token_mean"]
        pf = p["prefill_seconds_per_token"]
        A = s / m
        B = (pf + s) / (pf + m)
        eA = abs(A / raw - 1.0)
        worstA = max(worstA, eA)
        nA += 1
        print("%-16s %-9s %14.9f %14.9f %14.9f %10.2e"
              % (label, nm, raw, A, B, eA))
print("-" * len(hdr))
print("cells checked: %d   worst relative error of hypothesis A: %.3e" % (nA, worstA))
print()
if worstA < 1e-9:
    print("VERDICT: raw_ratio_of_means IS EXACTLY serial/mtp on every cell.")
    print("         prefill_seconds_per_token is reported but NOT scored.")
    print("         => prefill optimisation is worth EXACTLY ZERO to the score.")
