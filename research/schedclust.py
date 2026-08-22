"""Cluster board rows by their realised per-prompt draft-depth vector.

Every correct solver emits the same greedy token trajectory, so the only thing
that can differ is how many rounds it takes and what each round costs.  The
depth vector is therefore a fingerprint of the draft schedule.  Rows that share
a depth vector share a schedule generation.

For each cluster report: n, the depth vector, the median candidate seconds per
token per prompt, the implied published median, and the leading solvers.
"""
import collections
import json
import statistics as st

CACHE = "/tmp/yukon-board/full.json"
PROMPT_NAMES = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}
NAMES = ["plutarch", "drama", "travel", "beagle", "medicine",
         "essays", "republic", "botany"]
PINNED = "559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71"

raw = json.load(open(CACHE))
if isinstance(raw, dict):
    raw = raw["submissions"]

rows = []
serial_all = collections.defaultdict(list)
for r in raw:
    if not isinstance(r, dict):
        continue
    pp = (r.get("officialMetrics") or {}).get("per_prompt")
    if not pp or r.get("officialScore") is None:
        continue
    v = {}
    for e in pp:
        n = PROMPT_NAMES.get(e["prompt_sha256"][:8])
        if n:
            v[n] = e
            if e.get("serial_seconds_per_token_mean"):
                serial_all[n].append(e["serial_seconds_per_token_mean"])
    if len(v) != 8:
        continue
    if any(v[n].get("head_provenance_sha256") != PINNED for n in NAMES):
        continue
    if any(v[n].get("mtp_seconds_per_token_mean") is None
           or v[n].get("effective_mean_draft_len") is None for n in NAMES):
        continue
    rows.append((r, v))

smean = {n: st.mean(serial_all[n]) for n in NAMES}


def pub(times):
    ratios = sorted(smean[n] / times[n] for n in NAMES)
    return (ratios[3] + ratios[4]) / 2


groups = collections.defaultdict(list)
for r, v in rows:
    key = tuple(round(v[n]["effective_mean_draft_len"] * 4) / 4 for n in NAMES)
    groups[key].append((r, v))

print("pinned-head rows with a full 8-prompt vector: %d" % len(rows))
print("distinct depth-vector clusters: %d" % len(groups))
print()

big = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:8]
for key, members in big:
    times = {}
    for n in NAMES:
        times[n] = st.median(v[n]["mtp_seconds_per_token_mean"]
                             for _, v in members)
    scores = sorted(r.get("officialScore") for r, _ in members)
    who = collections.Counter(str(r.get("solverUsername")) for r, _ in members)
    print("n=%-4d depth %s" % (
        len(members), " ".join("%.2f" % x for x in key)))
    print("       med s/tok %s" % " ".join(
        "%.5f" % (1000 * times[n]) for n in NAMES))
    print("       implied published median (from cluster medians) = %.6f"
          % pub(times))
    print("       observed score: min %.5f  median %.5f  max %.5f" % (
        scores[0], st.median(scores), scores[-1]))
    print("       top solvers: %s" % ", ".join(
        "%s(%d)" % (w, c) for w, c in who.most_common(5)))
    print()

print("prompt order: %s" % " ".join(NAMES))
print()
print("BEST-OF-CLUSTERS per prompt (median of the best cluster for that prompt)")
best = {}
for n in NAMES:
    cand = []
    for key, members in groups.items():
        if len(members) < 5:
            continue
        cand.append((st.median(v[n]["mtp_seconds_per_token_mean"]
                               for _, v in members), key, len(members)))
    cand.sort()
    best[n] = cand[0][0]
    print("  %-9s %.8f  at depth %.2f  (cluster n=%d)" % (
        n, cand[0][0], cand[0][1][NAMES.index(n)], cand[0][2]))
print()
print("published median if one schedule hit every cluster best = %.6f" % pub(best))
print("crown 3.35922017 -> %+.3f %%" % (100 * (pub(best) / 3.35922017 - 1)))
