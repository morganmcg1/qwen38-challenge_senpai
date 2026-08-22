"""F92: what acceptance regime do the ranked hidden prompts actually run in?

For every frontier-class board row (officialScore >= 3.30) read the per-prompt
effective_mean_draft_len (drafts PROPOSED per round, F78) and the
non_drafting_round_count.  Decode length is fixed at 512 tokens.

  tokens_per_round = 512 / R
  accepted_drafts_per_round = tokens_per_round - 1
  accept_rate = accepted_drafts_per_round / effective_mean_draft_len

R is not published directly, but rounds = 512 / tokens_per_round and the F12
table already pinned R per prompt.  Here we recover R from the identity

  512 = R * (1 + accepted_per_round)

which needs accepted_per_round.  Instead use the published seconds:
  mtp_seconds_per_token_mean * 512 = total decode seconds
and F12's R.  Simpler and assumption-free: report the eff draft length
distribution, which is the quantity that separates the two regimes, plus the
non-drafting round count share.
"""
import collections
import json
import statistics as st

CACHE = "/tmp/yukon-board/full.json"
PROMPT_NAMES = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}
# F83 marginal weight on the published median (percentage points per 1% gain).
WEIGHT = {
    "beagle": 0.4862, "medicine": 0.2508, "essays": 0.1598,
    "botany": 0.0124, "republic": 0.0100,
    "plutarch": 0.0, "drama": 0.0, "travel": 0.0,
}
# F12 ranked round counts over the 512-token window.
ROUNDS = {
    "plutarch": 487, "drama": 252, "travel": 212, "beagle": 110,
    "republic": 93, "essays": 92, "medicine": 90, "botany": 81,
}

raw = json.load(open(CACHE))
if isinstance(raw, dict):
    raw = raw["submissions"]
rows = [r for r in raw
        if isinstance(r, dict)
        and (r.get("officialMetrics") or {}).get("per_prompt")
        and (r.get("officialScore") or 0) >= 3.30]

eff = collections.defaultdict(list)
nds = collections.defaultdict(list)
for r in rows:
    for e in r["officialMetrics"]["per_prompt"]:
        name = PROMPT_NAMES.get(e["prompt_sha256"][:8])
        if name is None:
            continue
        if e.get("effective_mean_draft_len") is not None:
            eff[name].append(e["effective_mean_draft_len"])
        if e.get("non_drafting_round_count") is not None:
            nds[name].append(e["non_drafting_round_count"])

print("frontier rows (officialScore >= 3.30): %d" % len(rows))
print()
print("%-10s %7s %7s %7s %8s %8s %8s %8s" % (
    "prompt", "weight", "R", "effmed", "tok/rnd", "acc/rnd", "accrate", "nondraft"))
for name in sorted(PROMPT_NAMES.values(), key=lambda n: -WEIGHT[n]):
    v = eff.get(name)
    if not v:
        continue
    m = st.median(v)
    R = ROUNDS[name]
    tpr = 512.0 / R
    apr = tpr - 1.0
    rate = apr / m if m > 0 else float("nan")
    nd = st.median(nds[name]) if nds.get(name) else float("nan")
    print("%-10s %7.4f %7d %7.3f %8.3f %8.3f %8.3f %8.0f" % (
        name, WEIGHT[name], R, m, tpr, apr, rate, nd))

wsum = sum(WEIGHT.values())
print()
print("weight on prompts with accept rate > 0.75:")
tot = 0.0
for name in PROMPT_NAMES.values():
    v = eff.get(name)
    if not v:
        continue
    m = st.median(v)
    rate = (512.0 / ROUNDS[name] - 1.0) / m if m > 0 else 0.0
    if rate > 0.75:
        tot += WEIGHT[name]
print("  %.4f of %.4f = %.1f%%" % (tot, wsum, 100 * tot / wsum))
