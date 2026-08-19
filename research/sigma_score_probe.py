#!/usr/bin/env python3
"""DIRECT measurement of sigma_score: find submissions of the SAME TREE.

Two rows sharing submissionCommitSha (or promotedSourceRef) are the same code
scored in two independent sessions. Their score difference is a direct draw from
the score-noise distribution -- no modelling, no fixture, no assumptions.

Also resolves the tension:
  - organizer paired sigma_score  = 0.078%   (6 sessions, unmodified tree)
  - my serial-draw upper bound    = 0.175%   (assumes serial noise does NOT pair away)
"""
import json
import statistics as st
from collections import defaultdict

NAMES = {"919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
         "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
         "ea82dcb5": "republic", "3b10cb4d": "travel"}
ORDER = ["plutarch", "drama", "travel", "beagle", "medicine", "essays", "republic", "botany"]

rows = json.load(open("/tmp/rows_live.json"))
good = [r for r in rows if isinstance(r.get("officialScore"), (int, float))
        and r.get("officialMetrics")
        and r["officialMetrics"].get("prompt_count") == 8]


def pp(r):
    return {NAMES[p["prompt_sha256"][:8]]: p for p in r["officialMetrics"]["per_prompt"]}


for field in ("submissionCommitSha", "promotedSourceRef",
              "officialMetrics.commit", "qwen_mtp_weights_hash"):
    by = defaultdict(list)
    for r in good:
        if field == "officialMetrics.commit":
            k = r["officialMetrics"].get("commit")
        elif field == "qwen_mtp_weights_hash":
            k = r["officialMetrics"].get("qwen_mtp_weights_hash")
        else:
            k = r.get(field)
        if k:
            by[k].append(r)
    dups = {k: v for k, v in by.items() if len(v) > 1}
    print(f"{field:<26} distinct={len(by):4d}  groups_with_repeats={len(dups):4d}"
          f"  rows_in_repeats={sum(len(v) for v in dups.values()):4d}")

print()
print("=" * 100)
print("REPEATED submissionCommitSha -> direct sigma_score draws")
by = defaultdict(list)
for r in good:
    if r.get("submissionCommitSha"):
        by[r["submissionCommitSha"]].append(r)
deltas = []
shown = 0
for k, v in sorted(by.items(), key=lambda t: -len(t[1])):
    if len(v) < 2:
        continue
    v.sort(key=lambda r: r["createdAt"])
    scores = [r["officialScore"] for r in v]
    if max(scores) < 2.5:
        continue
    rel = 100 * (max(scores) / min(scores) - 1)
    deltas.append((rel, k, v))
    if shown < 12:
        shown += 1
        print(f"  {k[:12]}  n={len(v)}  user={v[0].get('solverUsername')}"
              f"  spread={rel:.4f}%")
        for r in v:
            m = r["officialMetrics"]
            print(f"      {r['createdAt'][:19]}  {r['id'][:8]}  score {r['officialScore']:.8f}"
                  f"  serial {m['baseline_serial_seconds_per_token_mean']:.8f}"
                  f"  cand {m['candidate_mtp_seconds_per_token_mean']:.8f}  {r['status']}")

print()
if deltas:
    pairs = []
    for rel, k, v in deltas:
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                a, b = v[i]["officialScore"], v[j]["officialScore"]
                pairs.append(100 * abs(a - b) / ((a + b) / 2))
    print(f"  identical-tree score differences: n={len(pairs)}")
    print(f"    mean |delta| = {st.mean(pairs):.4f}%   max = {max(pairs):.4f}%")
    if len(pairs) > 1:
        # E|X-Y| = 2*sigma/sqrt(pi) for two iid normals
        print(f"    implied sigma_score = mean|d|*sqrt(pi)/2 = "
              f"{st.mean(pairs)*(3.14159265**0.5)/2:.4f}%")

print()
print("=" * 100)
print("CANDIDATE-LEG variability in the competitive cluster, per prompt")
print("If the ~0.23% serial noise were session common-mode, the candidate leg would")
print("show the same CV and pairing would cancel it. If the candidate CV is much")
print("smaller, the serial noise is per-measurement and does NOT pair away.")
comp = [r for r in good if r["officialScore"] > 3.20]
print(f"  n = {len(comp)}")
print(f"  {'prompt':<10} {'serial_CV':>10} {'cand_CV':>10} {'raw_p_CV':>10}  verdict")
for nm in ORDER:
    s = [pp(r)[nm]["serial_seconds_per_token_mean"] for r in comp]
    c = [pp(r)[nm]["mtp_seconds_per_token_mean"] for r in comp]
    p = [pp(r)[nm]["raw_ratio_of_means"] for r in comp]
    scv = 100 * st.stdev(s) / st.mean(s)
    ccv = 100 * st.stdev(c) / st.mean(c)
    pcv = 100 * st.stdev(p) / st.mean(p)
    verdict = "pairs away" if pcv < scv * 0.7 else "does NOT pair away"
    print(f"  {nm:<10} {scv:>9.3f}% {ccv:>9.3f}% {pcv:>9.3f}%  {verdict}")

print()
print("=" * 100)
print("SANITY: is raw_p exactly serial/cand per prompt? (no hidden normalisation)")
bad = 0
for r in good[:60]:
    for nm in ORDER:
        q = pp(r)[nm]
        lhs = q["raw_ratio_of_means"]
        rhs = q["serial_seconds_per_token_mean"] / q["mtp_seconds_per_token_mean"]
        if abs(lhs - rhs) > 1e-9 * max(1.0, abs(lhs)):
            bad += 1
print(f"  checked 60 rows x 8 prompts; mismatches = {bad}")
