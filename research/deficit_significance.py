#!/usr/bin/env python3
"""Test our per-prompt MTP deficit against the ORGANIZER-measured no-op spread.

Sources of truth:
  - our row  ca9251b8 (/tmp/ca92.json)
  - top row  0cd0a6b4 (fetched)
  - fixtures/qwen3_8_27b_mtp_track.json -> timed_prompt_pool[*].
      noop_decode_speedup_spread_pct  (conservative, over-states 1.4-1.9x)
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
# Exact pair-level ratios published in noop_decode_speedup_note.
EXACT_PAIR = {"beagle": 0.104, "botany": 0.281, "drama": 0.116}


def get(sub_id):
    tok = os.environ.get("YUKON_API_TOKEN", "")
    url = "https://api.yukon.org/api/submissions/" + sub_id
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read().decode())
    return d.get("submission", d)


fx = json.load(open("fixtures/qwen3_8_27b_mtp_track.json"))
spread = {}
for e in fx["timed_prompt_pool"]:
    nm = NAMES[e["sha256"][:8]]
    spread[nm] = e["noop_decode_speedup_spread_pct"]

ours = json.load(open("/tmp/ca92.json"))
ours = ours.get("submission", ours)

_live = json.load(open("/tmp/rows_live.json"))
_scored = [r for r in _live if isinstance(r.get("officialScore"), (int, float))]
_scored.sort(key=lambda r: -r["officialScore"])
TOP_ID = _scored[0]["id"]
print("top row id:", TOP_ID, "score", repr(_scored[0]["officialScore"]))
top = get(TOP_ID)


def bysha(sub):
    out = {}
    for p in sub["officialMetrics"]["per_prompt"]:
        out[NAMES[p["prompt_sha256"][:8]]] = p
    return out


A, B = bysha(ours), bysha(top)

print("pair counts (ours):", {k: v.get("accepted_pair_count") for k, v in A.items()})
print()
hdr = (
    "%-9s %6s %5s %9s %9s %8s %8s %8s %7s %6s"
    % ("prompt", "n", "pairs", "mtp_ours", "mtp_top", "d_pct", "d_ms512",
           "sigma%", "sigmas", "sig?")
)
print(hdr)
print("-" * len(hdr))
tot = 0.0
for nm in sorted(A, key=lambda k: A[k]["effective_mean_draft_len"]):
    a, b = A[nm], B[nm]
    ma, mb = a["mtp_seconds_per_token_mean"], b["mtp_seconds_per_token_mean"]
    dpct = (ma / mb - 1.0) * 100.0
    dms = (ma - mb) * 512 * 1000.0
    tot += dms
    # true paired sigma: exact where published, else conservative/1.65
    sig = EXACT_PAIR.get(nm, spread[nm] / 1.65)
    nsig = dpct / sig if sig else 0.0
    print(
        "%-9s %6.3f %5s %9.7f %9.7f %+8.3f %+8.1f %8.3f %+7.2f %6s"
        % (
            nm,
            a["effective_mean_draft_len"],
            a.get("accepted_pair_count"),
            ma,
            mb,
            dpct,
            dms,
            sig,
            nsig,
            "YES" if abs(nsig) >= 2 else "no",
        )
    )
print("-" * len(hdr))
print("total absolute deficit over the 8 windows: %+.1f ms" % tot)

print()
print("order-statistic saturation caps (our sorted raw ratios):")
rr = sorted(((A[nm]["raw_ratio_of_means"], nm) for nm in A))
for i, (v, nm) in enumerate(rr, 1):
    tag = ""
    if i == 4:
        tag = "  <-- 4th (central)"
    elif i == 5:
        tag = "  <-- 5th (central)"
    elif i == 6:
        tag = "  <-- 6th (caps the 5th)"
    print("  %d %-9s %.5f%s" % (i, nm, v, tag))
b4, b5, b6 = rr[3][0], rr[4][0], rr[5][0]
print("  score = mean(4th,5th) = %.8f" % ((b4 + b5) / 2))
print("  headroom on the 4th (%s) before it passes the 6th: %+.2f %%"
      % (rr[3][1], (b6 / b4 - 1) * 100))
print("  headroom on the 5th (%s) before it passes the 6th: %+.2f %%"
      % (rr[4][1], (b6 / b5 - 1) * 100))
