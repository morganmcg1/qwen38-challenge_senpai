#!/usr/bin/env python3
"""E89 cross-check: price the binary host-state mode on the real score statistic.

The advisor asked me to redo his replicate analysis from my own pipeline
(PR #90, feedback 5). Method, matching his definition exactly:

  1. Identify a submission's scored surface by the object ids that
     `git ls-tree <branch> Sources Vendor mtp-head.manifest.json` returns.
     Two submissions with the same three ids ran byte-identical scored code.
  2. Inside an identity group, keep run pairs whose eight
     effective_mean_draft_len values agree to 1e-3, so the pair really is a
     replicate and not a behaviour change.
  3. Split the pairs at a 0.7 % mean absolute gap over the seven drafting
     prompts, and check that the band around the split is empty.
  4. Price the mode on the published score statistic, which is the mean of the
     4th and 5th sorted per-prompt raw ratios, not on the seven-prompt mean.

Data provenance:
  curl -s -H "Authorization: Bearer $YUKON_API_TOKEN" \
    'https://api.yukon.org/api/benchmarks/5d1ee4d7-80bd-4555-b182-6505f26ef495/submissions?limit=2000' \
    | python3 -c 'import json,sys; json.dump(json.load(sys.stdin)["submissions"], open("/tmp/rows_live.json","w"))'
  git fetch --no-tags upstream

usage: research/e89_mode_price.py [JSON_OUT]
"""
import itertools
import json
import statistics as st
import subprocess
import sys

ROWS = "/tmp/rows_live.json"
FIXTURE = "fixtures/qwen3_8_27b_mtp_track.json"
IDENTITY_PATHS = ["Sources", "Vendor", "mtp-head.manifest.json"]
CONTROL = "plutarch"


def prompt_names():
    """Authoritative sha-to-name map, read from the track fixture.

    research/e40_width_tax_feasibility.py hardcodes this map with republic and
    botany transposed, so do not copy it from there.
    """
    out = {}
    for entry in json.load(open(FIXTURE))["timed_prompt_pool"]:
        source = next(v for v in entry.values()
                      if isinstance(v, str) and "qwen3.8-27b-pool-" in v)
        out[entry["sha256"][:8]] = source.split("qwen3.8-27b-pool-")[1].split(".")[0]
    return out


NAME = prompt_names()
SPLIT_PCT = 0.7
DRAFTLEN_TOL = 1e-3


def batch_check(specs):
    p = subprocess.run(["git", "cat-file", "--batch-check=%(objectname) %(rest)"],
                       input="".join(f"{s} {s}\n" for s in specs),
                       capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"git cat-file failed: {p.stderr.strip()}")
    out = {}
    for line in p.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] in ("missing", "ambiguous"):
            out[parts[0]] = None
        elif len(parts) >= 2:
            out[parts[1]] = parts[0]
    return out


def load_runs():
    runs = []
    for r in json.load(open(ROWS)):
        om = r.get("officialMetrics") or {}
        pp = om.get("per_prompt")
        if not isinstance(pp, list) or len(pp) != 8 or r.get("officialScore") is None:
            continue
        by = {}
        for entry in pp:
            key = NAME.get((entry.get("prompt_sha256") or "")[:8])
            if key is None:
                break
            by[key] = entry
        if len(by) != 8:
            continue
        runs.append({"id": r["id"], "score": r["officialScore"],
                     "created": r.get("createdAt"), "by": by})
    return runs


def score_statistic(run):
    """Published score = mean of the 4th and 5th sorted per-prompt raw ratios."""
    ratios = sorted(e["raw_ratio_of_means"] for e in run["by"].values())
    return (ratios[3] + ratios[4]) / 2.0


def main():
    out_json = sys.argv[1] if len(sys.argv) > 1 else None
    runs = load_runs()
    specs = [f"upstream/submissions/{r['id']}:{p}"
             for r in runs for p in IDENTITY_PATHS]
    oids = batch_check(specs)
    for r in runs:
        got = [oids.get(f"upstream/submissions/{r['id']}:{p}") for p in IDENTITY_PATHS]
        r["identity"] = tuple(got) if got[0] and got[1] else None

    resolved = [r for r in runs if r["identity"]]
    groups = {}
    for r in resolved:
        groups.setdefault(r["identity"], []).append(r)
    groups = {k: v for k, v in groups.items() if len(v) > 1}

    doc = {"scored_rows_with_metrics": len(runs),
           "rows_with_local_ref": len(resolved),
           "replicate_groups": len(groups),
           "runs_in_groups": sum(len(v) for v in groups.values())}

    drafting = [n for n in NAME.values() if n != CONTROL]
    pairs = []
    for members in groups.values():
        for a, b in itertools.combinations(members, 2):
            if any(abs(a["by"][n]["effective_mean_draft_len"]
                       - b["by"][n]["effective_mean_draft_len"]) > DRAFTLEN_TOL
                   for n in NAME.values()):
                continue
            slow, fast = a, b
            if (st.mean(a["by"][n]["mtp_seconds_per_token_mean"] for n in drafting)
                    < st.mean(b["by"][n]["mtp_seconds_per_token_mean"] for n in drafting)):
                slow, fast = b, a
            gap = {n: 100.0 * (slow["by"][n]["mtp_seconds_per_token_mean"]
                               - fast["by"][n]["mtp_seconds_per_token_mean"])
                   / fast["by"][n]["mtp_seconds_per_token_mean"]
                   for n in NAME.values()}
            serial = {n: 100.0 * (slow["by"][n]["serial_seconds_per_token_mean"]
                                  - fast["by"][n]["serial_seconds_per_token_mean"])
                      / fast["by"][n]["serial_seconds_per_token_mean"]
                      for n in NAME.values()}
            s_slow, s_fast = score_statistic(slow), score_statistic(fast)
            pairs.append({
                "slow": slow["id"], "fast": fast["id"], "gap": gap,
                "serial_gap": serial,
                "mean7_abs": st.mean(abs(gap[n]) for n in drafting),
                "mean7": st.mean(gap[n] for n in drafting),
                "candidate_two_prompt": st.mean([gap["beagle"], gap["essays"]]),
                "score_slow": s_slow, "score_fast": s_fast,
                "score_gain_pct": 100.0 * (s_fast - s_slow) / s_slow,
                "score_gap_abs": abs(s_fast - s_slow)})

    cross = [p for p in pairs if p["mean7_abs"] >= SPLIT_PCT]
    same = [p for p in pairs if p["mean7_abs"] < SPLIT_PCT]
    doc["pairs"] = len(pairs)
    doc["cross_mode_pairs"] = len(cross)
    doc["same_mode_pairs"] = len(same)

    print(f"scored rows with metrics {len(runs)}, with a local submission ref "
          f"{len(resolved)}")
    print(f"replicate groups {len(groups)}, runs in groups "
          f"{sum(len(v) for v in groups.values())}, draft-length-matched pairs "
          f"{len(pairs)}")

    ordered = sorted(p["mean7_abs"] for p in pairs)
    doc["mean7_abs_sorted"] = ordered
    below = [v for v in ordered if v < SPLIT_PCT]
    above = [v for v in ordered if v >= SPLIT_PCT]
    print("\nIS THE BAND EMPTY? sorted mean absolute gap over the 7 drafting prompts")
    if below and above:
        doc["band"] = {"highest_same_mode": below[-1], "lowest_cross_mode": above[0]}
        print(f"  same mode {len(below)} pairs, {below[0]:.3f} to {below[-1]:.3f} %")
        print(f"  cross mode {len(above)} pairs, {above[0]:.3f} to {above[-1]:.3f} %")
        print(f"  empty band {below[-1]:.3f} to {above[0]:.3f} %")

    print("\nCROSS-MODE PER-PROMPT PREMIUM, slow minus fast, median over pairs")
    doc["per_prompt_median_pct"] = {}
    doc["per_prompt_serial_median_pct"] = {}
    for n in NAME.values():
        m = st.median(p["gap"][n] for p in cross)
        s = st.median(p["serial_gap"][n] for p in cross)
        doc["per_prompt_median_pct"][n] = m
        doc["per_prompt_serial_median_pct"][n] = s
        print(f"  {n:<9} candidate {m:+7.3f} %   serial {s:+7.3f} %")

    print("\nPRICE OF THE MODE")
    mean7 = st.median(p["mean7"] for p in cross)
    two = st.median(p["candidate_two_prompt"] for p in cross)
    gain = st.median(p["score_gain_pct"] for p in cross)
    doc["mean7_median_pct"] = mean7
    doc["candidate_two_prompt_median_pct"] = two
    doc["score_statistic_gain_pct"] = gain
    doc["ratio_score_to_mean7"] = two / mean7
    doc["cross_mode_median_score_gap_abs"] = st.median(p["score_gap_abs"] for p in cross)
    print(f"  mean7, candidate side                       {mean7:+.3f} %")
    print(f"  beagle and essays, candidate side           {two:+.3f} %")
    print(f"  ratio of the two-prompt gap to mean7        {two / mean7:.3f}")
    print(f"  SCORE-STATISTIC GAIN FROM REMOVING THE MODE {gain:+.3f} %")
    print(f"  median published-score gap, cross mode      "
          f"{st.median(p['score_gap_abs'] for p in cross):.4f}")
    if same:
        doc["same_mode_median_score_gap_abs"] = st.median(p["score_gap_abs"] for p in same)
        rep = sorted(abs(p["score_gain_pct"]) for p in same)
        doc["same_mode_reproducibility_median_pct"] = st.median(rep)
        doc["same_mode_reproducibility_max_pct"] = rep[-1]
        print(f"  median published-score gap, same mode       "
              f"{st.median(p['score_gap_abs'] for p in same):.4f}")
        print(f"  same-mode reproducibility on the score      "
              f"median {st.median(rep):.3f} %, max {rep[-1]:.3f} %")

    print("\nDOES THE SCORE STATISTIC REPRODUCE officialScore?")
    err = sorted(abs(score_statistic(r) - r["score"]) for r in runs)
    doc["score_statistic_max_abs_error"] = err[-1]
    print(f"  max absolute error over {len(runs)} scored rows {err[-1]:.3e}")
    def setters(run):
        s = sorted(run["by"].items(), key=lambda kv: kv[1]["raw_ratio_of_means"])
        return s[3][0], s[4][0]

    print("\nWHICH PROMPTS SET THE SCORE, by score band")
    doc["score_setting_prompt_pairs"] = {}
    ranked = sorted(runs, key=lambda r: -r["score"])
    for n in (10, 25, 50, 100, len(ranked)):
        which = {}
        for r in ranked[:n]:
            which[setters(r)] = which.get(setters(r), 0) + 1
        top = sorted(which.items(), key=lambda kv: -kv[1])[:3]
        label = "all" if n == len(ranked) else f"top{n}"
        doc["score_setting_prompt_pairs"][label] = {f"{a}+{b}": c for (a, b), c in top}
        print(f"  {label:<6} n={n:<4} " +
              ", ".join(f"{a}+{b} {c}" for (a, b), c in top))

    changed = sum(1 for p in cross
                  if setters(next(r for r in runs if r["id"] == p["slow"]))
                  != setters(next(r for r in runs if r["id"] == p["fast"])))
    doc["cross_mode_pairs_whose_score_setters_change"] = changed
    print(f"  cross-mode pairs where the mode changes which prompts set the "
          f"score: {changed} of {len(cross)}")

    if out_json:
        json.dump(doc, open(out_json, "w"), indent=2, sort_keys=True)
        print(f"\nwrote {out_json}")


main()
