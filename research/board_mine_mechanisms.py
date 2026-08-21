"""Mine the whole official board for schedule-matched mechanism measurements.

Why this exists
---------------
The published score is `median_p( serial_p / candidate_p )`. Finding 20 showed
that the serial denominator is an independent per-run draw with a published
resample floor near 0.277 %, which is larger than almost every real mechanism
anyone has shipped. So a board rank is mostly a lottery ticket and tells you
very little about engineering.

The candidate leg is different. `per_prompt[p].mtp_seconds_per_token_mean` is
the candidate's own decode time, and it is not touched by the serial draw. If
two runs executed the SAME draft schedule, then round for round they did the
same work, and the difference in their candidate legs is a clean measurement of
whatever code differs between them.

`effective_mean_draft_len` is reported to full double precision on every prompt.
Two runs whose eight values are bit-identical ran the same trajectory. That is
the matching key used here.

What it reports
---------------
1. COHORTS. Every group of runs sharing one bit-identical eight-prompt
   schedule signature. Inside a cohort the schedule is not a confounder.

2. NULLS. Inside a cohort, runs that share a `submissionCommitSha` are
   byte-identical replicates. Their candidate-leg spread is a MEASURED noise
   floor for that cohort, with no modelling.

3. MECHANISMS. Inside a cohort, every pair of distinct commits, with the
   candidate-leg contrast in percent, its per-prompt standard deviation, the
   sign count, and a z score against the cohort null when one exists.

4. PROMPT RESOLUTION. Per-prompt candidate-leg standard deviation inside each
   large cohort. This is what makes an instrument: a prompt with a small sd
   resolves a small mechanism. Finding 29b came out of this column.

Reading the sign
----------------
A POSITIVE `cand%` means B decoded FASTER than A, because the score is a ratio
with the candidate time in the denominator. The same convention is used by
`research/board_pair_decompose.py`.

What it cannot tell you
-----------------------
It measures the difference between two trees. It does not tell you WHICH source
change caused it. Take the two `submissionCommitSha` values and diff the
matching `upstream/submissions/*` branches to name the mechanism, exactly as
the campaign ledger does.

A cohort match also proves the schedule was identical, which is strong evidence
that no drafting-policy change is in the diff. A run that changes acceptance or
depth policy leaves its own cohort and cannot be measured this way at all.

Usage
-----
    python3 research/board_mine_mechanisms.py                 # summary
    python3 research/board_mine_mechanisms.py --min-score 3.2 # restrict
    python3 research/board_mine_mechanisms.py --cohort <sig>  # one cohort
    python3 research/board_mine_mechanisms.py --prompts       # resolution table

Reads /tmp/yukon-board/full.json, or the path in $YUKON_BOARD_JSON.
"""

import argparse
import json
import math
import os
import statistics
import sys

BOARD_JSON = os.environ.get("YUKON_BOARD_JSON", "/tmp/yukon-board/full.json")

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
PROMPT_ORDER = ["plutarch", "drama", "travel", "beagle", "medicine", "republic",
                "essays", "botany"]


def load_rows(path=BOARD_JSON):
    with open(path) as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        for key in ("submissions", "rows", "data", "items"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise SystemExit(f"unexpected board payload shape in {path}")
    return [r for r in payload if isinstance(r, dict)]


def prompt_map(row):
    """Return {prompt_name: per_prompt_dict} or None when the row is unusable."""
    metrics = row.get("officialMetrics") or {}
    per_prompt = metrics.get("per_prompt")
    if not isinstance(per_prompt, list) or len(per_prompt) != 8:
        return None
    out = {}
    for entry in per_prompt:
        if not isinstance(entry, dict):
            return None
        sha = (entry.get("prompt_sha256") or "")[:8]
        name = PROMPT_NAMES.get(sha)
        if name is None:
            return None
        if entry.get("mtp_seconds_per_token_mean") in (None, 0):
            return None
        if entry.get("effective_mean_draft_len") is None:
            return None
        out[name] = entry
    return out if len(out) == 8 else None


def schedule_signature(pmap):
    """Bit-exact eight-prompt draft schedule key.

    repr() of a float round-trips exactly in CPython, so this is an identity
    test on the schedule, not a tolerance test.
    """
    return tuple(repr(pmap[n]["effective_mean_draft_len"]) for n in PROMPT_ORDER)


def candidate_contrast(pmap_a, pmap_b):
    """Per-prompt percent by which B's candidate leg is faster than A's."""
    return {n: 100.0 * math.log(pmap_a[n]["mtp_seconds_per_token_mean"]
                                / pmap_b[n]["mtp_seconds_per_token_mean"])
            for n in PROMPT_ORDER}


def serial_contrast(pmap_a, pmap_b):
    return {n: 100.0 * math.log(pmap_a[n]["serial_seconds_per_token_mean"]
                                / pmap_b[n]["serial_seconds_per_token_mean"])
            for n in PROMPT_ORDER}


def summarise(values):
    vals = [values[n] for n in PROMPT_ORDER]
    mean = statistics.fmean(vals)
    sd = statistics.stdev(vals)
    positive = sum(1 for v in vals if v > 0)
    return mean, sd, positive


def build_cohorts(rows, min_score):
    cohorts = {}
    for row in rows:
        score = row.get("officialScore")
        if score is None or score < min_score:
            continue
        pmap = prompt_map(row)
        if pmap is None:
            continue
        sig = schedule_signature(pmap)
        cohorts.setdefault(sig, []).append((row, pmap))
    return cohorts


def cohort_null(members):
    """Measured candidate-leg noise floor from byte-identical replicate pairs.

    Returns (sd_percent, n_pairs) or (None, 0). Each pair contributes the mean
    contrast between two runs of the SAME commit, which is by construction a
    draw from the zero-mechanism distribution.
    """
    by_commit = {}
    for row, pmap in members:
        by_commit.setdefault(row.get("submissionCommitSha"), []).append(pmap)
    deltas = []
    for commit, maps in by_commit.items():
        if commit is None or len(maps) < 2:
            continue
        for i in range(len(maps)):
            for j in range(i + 1, len(maps)):
                deltas.append(summarise(candidate_contrast(maps[i], maps[j]))[0])
    if len(deltas) < 2:
        return None, len(deltas)
    # Mean is zero by symmetry, so use the raw second moment.
    return math.sqrt(sum(d * d for d in deltas) / len(deltas)), len(deltas)


def short(row):
    return (row.get("id") or "")[:8]


def report_cohorts(cohorts, min_members, limit):
    ranked = sorted(cohorts.items(), key=lambda kv: -len(kv[1]))
    print(f"{'cohort':>8}  {'runs':>4}  {'commits':>7}  {'meandl':>6}  "
          f"{'nullsd%':>7}  {'npair':>5}  members")
    shown = 0
    for sig, members in ranked:
        if len(members) < min_members:
            continue
        commits = {row.get("submissionCommitSha") for row, _ in members}
        mean_dl = statistics.fmean(float(v) for v in sig)
        sd, npair = cohort_null(members)
        sd_text = f"{sd:7.4f}" if sd is not None else "      -"
        sample = " ".join(short(row) for row, _ in members[:6])
        if len(members) > 6:
            sample += f" +{len(members) - 6}"
        print(f"{abs(hash(sig)) % 10**8:08d}  {len(members):4d}  "
              f"{len(commits):7d}  {mean_dl:6.3f}  {sd_text}  {npair:5d}  {sample}")
        shown += 1
        if shown >= limit:
            break


def prompt_resolution(members):
    """Per-prompt candidate-leg sd inside one cohort, in percent."""
    out = {}
    for name in PROMPT_ORDER:
        logs = [100.0 * math.log(pmap[name]["mtp_seconds_per_token_mean"])
                for _, pmap in members]
        if len(logs) > 2:
            out[name] = statistics.stdev(logs)
    return out


# Plutarch draws only 38 drafting rounds out of 487, so its candidate leg is
# almost pure target decode. The five G=2 prompts draft on nearly every round.
# Contrasting the two isolates where a mechanism actually acts.
TARGET_PROBE = "plutarch"
DRAFT_PROBES = ["beagle", "medicine", "republic", "essays", "botany"]


def report_mechanisms(cohorts, min_members, min_effect, limit):
    """Every distinct-commit pair inside every cohort, strongest first.

    `plut%` is the target-path estimate and `draft%` is the drafting-path
    estimate. A mechanism in the target runtime moves both. A mechanism in the
    proposal head, the selection chain or the draft schedule moves `draft%`
    and leaves `plut%` at zero. `sigma` divides `plut%` by the cohort's own
    measured plutarch resolution, so it says how many resolution units of
    target-path evidence this single pair carries.
    """
    findings = []
    for sig, members in cohorts.items():
        if len(members) < min_members:
            continue
        resolution = prompt_resolution(members)
        plut_sd = resolution.get(TARGET_PROBE)
        best = {}
        for row, pmap in members:
            commit = row.get("submissionCommitSha")
            if commit is None:
                continue
            best.setdefault(commit, (row, pmap))
        commits = sorted(best)
        for i in range(len(commits)):
            for j in range(i + 1, len(commits)):
                row_a, pmap_a = best[commits[i]]
                row_b, pmap_b = best[commits[j]]
                contrast = candidate_contrast(pmap_a, pmap_b)
                mean, sd, positive = summarise(contrast)
                if abs(mean) < min_effect:
                    continue
                s_mean, _, _ = summarise(serial_contrast(pmap_a, pmap_b))
                plut = contrast[TARGET_PROBE]
                draft = statistics.fmean(contrast[n] for n in DRAFT_PROBES)
                sigma = plut / plut_sd if plut_sd else float("nan")
                findings.append((abs(mean), short(row_a), short(row_b), mean, sd,
                                 positive, s_mean, plut, draft, sigma,
                                 len(members)))
    findings.sort(reverse=True)
    print(f"{'A':>8} {'B':>8} {'cand%':>8} {'sd':>7} {'+/8':>4} "
          f"{'serial%':>8} {'plut%':>8} {'draft%':>8} {'sigma':>7} {'cohort':>6}")
    for (_, a, b, mean, sd, positive, s_mean, plut, draft, sigma,
         n) in findings[:limit]:
        sigma_text = f"{sigma:7.2f}" if sigma == sigma else "      -"
        print(f"{a:>8} {b:>8} {mean:+8.4f} {sd:7.4f} {positive:4d} "
              f"{s_mean:+8.4f} {plut:+8.4f} {draft:+8.4f} {sigma_text} {n:6d}")
    print(f"\n{len(findings)} distinct-commit pairs at |cand%| >= {min_effect}")


def report_prompts(cohorts, min_members):
    """Per-prompt candidate-leg resolution, pooled over large cohorts."""
    pooled = {n: [] for n in PROMPT_ORDER}
    cohorts_used = 0
    for sig, members in cohorts.items():
        if len(members) < min_members:
            continue
        cohorts_used += 1
        for name in PROMPT_ORDER:
            logs = [100.0 * math.log(pmap[name]["mtp_seconds_per_token_mean"])
                    for _, pmap in members]
            centre = statistics.fmean(logs)
            pooled[name].extend(v - centre for v in logs)
    print(f"pooled over {cohorts_used} cohorts of >= {min_members} runs\n")
    print(f"{'prompt':>9} {'n':>5} {'sd%':>8}  resolution")
    rows = []
    for name in PROMPT_ORDER:
        vals = pooled[name]
        if len(vals) < 3:
            continue
        sd = math.sqrt(sum(v * v for v in vals) / (len(vals) - 1))
        rows.append((sd, name, len(vals)))
    rows.sort()
    if not rows:
        print("  no cohort large enough")
        return
    finest = rows[0][0]
    for sd, name, n in rows:
        bar = "#" * max(1, int(round(sd / finest)))
        print(f"{name:>9} {n:5d} {sd:8.4f}  {bar}")
    print(f"\nsharpest prompt: {rows[0][1]} at {finest:.4f} % candidate-leg sd")
    print("A single official receipt resolves a mechanism of that size on that")
    print("prompt alone, against a published-median floor of about 0.277 %.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--board", default=BOARD_JSON)
    parser.add_argument("--min-score", type=float, default=3.15,
                        help="ignore rows below this official score")
    parser.add_argument("--min-members", type=int, default=3,
                        help="smallest cohort to report")
    parser.add_argument("--min-effect", type=float, default=0.05,
                        help="smallest |candidate-leg percent| to list")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--mechanisms", action="store_true")
    parser.add_argument("--prompts", action="store_true")
    args = parser.parse_args(argv)

    rows = load_rows(args.board)
    cohorts = build_cohorts(rows, args.min_score)
    scored = sum(len(v) for v in cohorts.values())
    print(f"{len(rows)} board rows, {scored} scored at >= {args.min_score}, "
          f"{len(cohorts)} distinct schedules\n")

    if args.prompts:
        report_prompts(cohorts, args.min_members)
    elif args.mechanisms:
        report_mechanisms(cohorts, args.min_members, args.min_effect, args.limit)
    else:
        report_cohorts(cohorts, args.min_members, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
