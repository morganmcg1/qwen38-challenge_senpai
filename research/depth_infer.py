#!/usr/bin/env python3
"""Constrain realised draft depth from exposed per-prompt fields.

Exposed: effective_mean_draft_len (n), non_drafting_round_count (R0),
         accepted_pair_count.
Window : 512 decoded tokens per prompt (decodeTokens=512).

If a round emits (draft_accepted + 1) tokens, then
    sum over rounds of (a_i + 1) = 512
and n = mean(a_i) over ALL rounds  =>  R*(n+1) = 512  =>  R = 512/(n+1).
A non-drafting round has a_i = 0. So the mean over DRAFTING rounds is
    n_draft = n * R / (R - R0).
"""
import json
import os
import urllib.request

NAMES = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
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
top = get(scored[0]["id"])

for label, sub in (("OURS ca9251b8", ours), ("TOP  " + scored[0]["id"][:8], top)):
    print("=== " + label + " ===")
    hdr = "%-9s %8s %8s %8s %9s %9s %9s" % (
        "prompt", "n", "R0", "R_est", "R_draft", "n_draft", "R0/R %")
    print(hdr)
    print("-" * len(hdr))
    for p in sorted(
        sub["officialMetrics"]["per_prompt"],
        key=lambda p: p["effective_mean_draft_len"],
    ):
        nm = NAMES[p["prompt_sha256"][:8]]
        n = p["effective_mean_draft_len"]
        r0 = p.get("non_drafting_round_count")
        R = 512.0 / (n + 1.0)
        rd = R - (r0 or 0)
        nd = (n * R / rd) if rd > 0 else float("nan")
        print(
            "%-9s %8.4f %8s %8.1f %9.1f %9.4f %9.1f"
            % (nm, n, r0, R, rd, nd, 100.0 * (r0 or 0) / R)
        )
    print()
