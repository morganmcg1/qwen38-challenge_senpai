"""Print the F76 absolute mode index for named submissions, as anchors.

index = sum_p w_p * 100 * ln(mtp_seconds_per_token_mean_p), weights sum to 0.
One mode flip is 1.000 index units; same-mode sd is 0.116 and per-run noise
is 0.0817, so a 1.0 gap is 8.6 same-mode sigmas.
"""
import json
import math
import sys

CACHE = "/tmp/yukon-board/full.json"
PROMPT_NAMES = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}
W = {"plutarch": -0.3852, "drama": +0.0215, "travel": +0.4945,
     "beagle": +0.2068, "medicine": -0.1480, "republic": -0.0917,
     "essays": -0.0041, "botany": -0.0939}

raw = json.load(open(CACHE))
if isinstance(raw, dict):
    raw = raw["submissions"]

want = sys.argv[1:] or [
    "f04b102e", "87b654b2", "b8b8b860", "44559d02", "7bef7d4c", "cb8aeefb",
    "bc070b7b", "7358c89f", "51b9bf85", "276aa2c2", "8819b108", "cf9a9eda",
]

rows = {}
for r in raw:
    if not isinstance(r, dict):
        continue
    sid = str(r.get("id") or "")
    for w in want:
        if sid.startswith(w):
            rows[w] = r

print("%-10s %-16s %10s %12s %8s %s" % (
    "id", "solver", "score", "mode index", "status", "resolvedAt"))
for w in want:
    r = rows.get(w)
    if r is None:
        print("%-10s  NOT ON THE BOARD YET" % w)
        continue
    pp = (r.get("officialMetrics") or {}).get("per_prompt")
    if not pp:
        print("%-10s %-16s %10s  no per-prompt rows  %s" % (
            w, str(r.get("solverUsername"))[:16],
            r.get("officialScore"), r.get("status")))
        continue
    idx = 0.0
    for e in pp:
        n = PROMPT_NAMES.get(e["prompt_sha256"][:8])
        t = e.get("mtp_seconds_per_token_mean")
        if n and t:
            idx += W[n] * 100.0 * math.log(t)
    print("%-10s %-16s %10s %12.4f %8s %s" % (
        w, str(r.get("solverUsername"))[:16],
        ("%.8f" % r["officialScore"]) if r.get("officialScore") else "-",
        idx, r.get("status"), r.get("resolvedAt") or r.get("createdAt")))
