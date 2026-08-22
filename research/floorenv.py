"""What published median would the union of every solver's best per-prompt
candidate time produce?  Recompute the median exactly over all eight prompts.

Also report the same envelope built from the 5th-best and 10th-best per-prompt
times, because a minimum over 782 noisy draws is biased low by the per-leg
noise (E112 pooled per-leg sd 0.3647 %).
"""
import json
import statistics as st

CACHE = "/tmp/yukon-board/full.json"
PROMPT_NAMES = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}
NAMES = list(PROMPT_NAMES.values())
PINNED = "559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71"

raw = json.load(open(CACHE))
if isinstance(raw, dict):
    raw = raw["submissions"]
rows = [r for r in raw
        if isinstance(r, dict)
        and (r.get("officialMetrics") or {}).get("per_prompt")
        and r.get("officialScore") is not None]

cand = {n: [] for n in NAMES}
serial = {n: [] for n in NAMES}
own = {n: [] for n in NAMES}
for r in rows:
    mine = r.get("solverUsername") == "morganmcg1"
    for e in r["officialMetrics"]["per_prompt"]:
        n = PROMPT_NAMES.get(e["prompt_sha256"][:8])
        if n is None:
            continue
        t = e.get("mtp_seconds_per_token_mean")
        s = e.get("serial_seconds_per_token_mean")
        if t:
            cand[n].append((t, r.get("solverUsername"), r.get("officialScore")))
            if mine:
                own[n].append(t)
        if s:
            serial[n].append(s)

smean = {n: st.mean(serial[n]) for n in NAMES}


def median8(times):
    ratios = sorted(smean[n] / times[n] for n in NAMES)
    return (ratios[3] + ratios[4]) / 2, ratios


def envelope(k):
    return {n: sorted(t for t, _, _ in cand[n])[k] for n in NAMES}


print("prompt      n   best        5th         10th        25th        ours-best   "
      "ours-med")
for n in NAMES:
    v = sorted(t for t, _, _ in cand[n])
    o = sorted(own[n])
    print("%-10s %4d %.8f %.8f %.8f %.8f %.8f %.8f" % (
        n, len(v), v[0], v[4], v[9], v[24],
        o[0] if o else float("nan"),
        st.median(o) if o else float("nan")))

print()
for k, label in ((0, "absolute floor"), (4, "5th best"),
                 (9, "10th best"), (24, "25th best")):
    m, ratios = median8(envelope(k))
    print("%-16s published median = %.6f   sorted ratios = %s" % (
        label, m, " ".join("%.3f" % x for x in ratios)))

print()
ourbest = {n: min(own[n]) for n in NAMES if own[n]}
if len(ourbest) == 8:
    m, ratios = median8(ourbest)
    print("%-16s published median = %.6f   sorted ratios = %s" % (
        "our own floors", m, " ".join("%.3f" % x for x in ratios)))

crown = 3.35922017
for k, label in ((0, "absolute floor"), (4, "5th best"), (9, "10th best")):
    m, _ = median8(envelope(k))
    print("%-16s headroom over crown %.5f = %+.3f %%" % (
        label, crown, 100 * (m / crown - 1)))
