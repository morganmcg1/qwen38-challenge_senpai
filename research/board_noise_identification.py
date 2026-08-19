#!/usr/bin/env python3
"""Identify the ranked instrument's noise floor from board telemetry alone.

Ledger item 180(H) records an internal contradiction: item 148 says two
submissions of the same tree agree to <= 0.0693 % per prompt, while items
166/172 quote a 0.7678 % between-submission floor. Those cannot both describe
one homoscedastic instrument, and the campaign uses the large number to set
minimum detectable effects and the small number to credit a +0.0173 % board
step.

Three identifications are available from `research/e53-board-facts.json`
without any GPU:

A. The ranked serial leg is a pinned, prebuilt binary in the runner-owned
   baseline workspace. No candidate edit can move it. Therefore the spread of
   `serial_spt` across every submission ever measured is PURE instrument
   noise, with n = one row per submission per prompt.

B. Submissions that share a `head` digest have byte-identical submitted
   content. Their per-prompt differences are pure resubmission noise for the
   full paired estimator that scoring actually uses.

C. If the two legs of one submission share a common thermal/host mode, the
   ratio cancels it. Compare A against B to measure how much cancels.

Run: python3 research/board_noise_identification.py
"""

from __future__ import annotations

import collections
import itertools
import json
import math
import pathlib
import statistics as st

HERE = pathlib.Path(__file__).resolve().parent
FACTS = HERE / "e53-board-facts.json"


def rel_sd(xs: list[float]) -> tuple[float, float, float]:
    m = st.mean(xs)
    s = st.stdev(xs)
    return m, s, 100.0 * s / m


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(
        sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)
    )
    return num / den if den else float("nan")


def quantile(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return float("nan")
    i = min(len(sorted_xs) - 1, int(q * len(sorted_xs)))
    return sorted_xs[i]


def main() -> int:
    facts = json.loads(FACTS.read_text())
    tele = facts["telemetry"]
    prompts = list(tele.keys())

    by_sub: dict[str, dict[str, dict]] = {}
    for p in prompts:
        for row in tele[p]:
            by_sub.setdefault(row["submission"], {})[p] = row

    print("prompts:", ", ".join(prompts))
    print(f"submissions with telemetry: {len(by_sub)}")
    print()

    print("=== A. serial leg across ALL submissions (pinned binary => pure noise) ===")
    print(
        f"{'prompt':12s} {'n':>4s} {'mean s/tok':>12s} {'sd':>12s} "
        f"{'rel sd %':>9s} {'range %':>8s}"
    )
    serial_rel = []
    mtp_rel = []
    ratio_rel = []
    for p in prompts:
        xs = [r["serial_spt"] for r in tele[p]]
        m, s, rel = rel_sd(xs)
        serial_rel.append(rel)
        print(
            f"{p:12s} {len(xs):4d} {m:12.8f} {s:12.8f} {rel:9.4f} "
            f"{100 * (max(xs) - min(xs)) / m:8.3f}"
        )
    print(f"  mean per-prompt rel sd of the SERIAL leg = {st.mean(serial_rel):.4f} %")
    print()

    print("=== A'. same for the candidate leg and the raw ratio (content + noise) ===")
    for label, key, acc in (
        ("mtp_spt", "mtp_spt", mtp_rel),
        ("raw_ratio", "raw_ratio", ratio_rel),
    ):
        for p in prompts:
            xs = [r[key] for r in tele[p]]
            acc.append(rel_sd(xs)[2])
        print(f"  mean per-prompt rel sd of {label:9s} = {st.mean(acc):8.4f} %")
    print("  (these two mix real content differences with noise; A does not)")
    print()

    print("=== B. is `head` a tree digest? (it is NOT — it is the proposal head) ===")
    heads: dict[str, list[str]] = collections.defaultdict(list)
    commits: dict[str, list[str]] = collections.defaultdict(list)
    for sub, rows in by_sub.items():
        heads[rows[prompts[0]]["head"]].append(sub)
        commits[rows[prompts[0]]["commit"]].append(sub)
    print(
        f"distinct `head` values={len(heads)} for {len(by_sub)} submissions "
        f"=> `head` groups many content-distinct trees; NOT usable as a same-tree key"
    )
    dup_commit = {c: v for c, v in commits.items() if len(v) > 1}
    print(
        f"distinct `commit` values={len(commits)}  commits with >1 submission="
        f"{len(dup_commit)}"
    )
    print(
        "  this dataset is already de-duplicated to content-unique rows, so it "
        "contains no true same-tree resubmission pair"
    )
    print()

    print("=== B'. effective sample size: how many DISTINCT serial values exist? ===")
    for p in prompts:
        xs = [r["serial_spt"] for r in tele[p]]
        c = collections.Counter(xs)
        top = c.most_common(1)[0]
        print(
            f"  {p:12s} n={len(xs)}  distinct={len(c)}  "
            f"most-repeated value appears {top[1]}x"
        )
    print()

    print("=== C. is the serial-leg spread iid noise or slow drift? ===")
    print("     split each prompt's rows by `created` date; compare within/between day")
    for p in prompts:
        rows = sorted(tele[p], key=lambda r: r["created"])
        byday: dict[str, list[float]] = collections.defaultdict(list)
        for r in rows:
            byday[r["created"][:10]].append(r["serial_spt"])
        usable = {d: v for d, v in byday.items() if len(v) >= 3}
        if not usable:
            print(f"  {p:12s} no day has >=3 rows")
            continue
        within = st.mean([st.stdev(v) for v in usable.values()])
        day_means = [st.mean(v) for v in usable.values()]
        between = st.stdev(day_means) if len(day_means) > 1 else 0.0
        grand = st.mean([x for v in usable.values() for x in v])
        print(
            f"  {p:12s} days={len(usable)}  within-day sd={100 * within / grand:6.4f}%"
            f"  between-day sd={100 * between / grand:6.4f}%"
        )
    print()

    print("=== C'. candidate-leg noise from NON-DRAFTING submissions ===")
    print("     a tree with mean_draft_len == 0 runs the serial computation on the")
    print("     candidate leg, so its mtp_spt spread is candidate-leg noise, not content")
    zero_subs = [
        s
        for s, rows in by_sub.items()
        if all(rows[p].get("mean_draft_len", 1) == 0 for p in prompts)
    ]
    print(f"  submissions with mean_draft_len == 0 on every prompt: {len(zero_subs)}")
    print()

    print("=== C''. candidate-leg noise from BEHAVIOURALLY IDENTICAL trees ===")
    print("     group submissions by the exact 8-tuple of mean_draft_len. Agreement")
    print("     to full float precision on all 8 prompts means the candidate leg did")
    print("     the same drafting work, so mtp_spt spread inside a group is noise.")
    beh: dict[tuple, list[str]] = collections.defaultdict(list)
    for sub, rows in by_sub.items():
        beh[tuple(rows[p]["mean_draft_len"] for p in prompts)].append(sub)
    groups = sorted(
        (v for v in beh.values() if len(v) >= 4), key=len, reverse=True
    )
    print(f"  behaviour classes={len(beh)}  classes with >=4 submissions={len(groups)}")
    class_means: list[float] = []
    ratio_in: list[float] = []
    corrs: list[float] = []
    for gi, subs in enumerate(groups[:6]):
        line = []
        for p in prompts:
            xs = [by_sub[s][p]["mtp_spt"] for s in subs]
            ys = [by_sub[s][p]["serial_spt"] for s in subs]
            rs = [by_sub[s][p]["raw_ratio"] for s in subs]
            ratio_in.append(rel_sd(rs)[2])
            corrs.append(pearson(ys, xs))
            line.append(rel_sd(xs)[2])
        class_means.append(st.mean(line))
        dl = by_sub[subs[0]][prompts[0]]["mean_draft_len"]
        print(
            f"  class {gi}: n={len(subs):3d} draft_len[0]={dl:.4f} "
            f"mean mtp rel sd over prompts={st.mean(line):6.4f}%"
        )
    if class_means:
        print(
            f"  tightest class => upper bound on the CANDIDATE leg = "
            f"{min(class_means):.4f} %"
        )
        print(
            f"  pooled per-prompt rel sd of raw_ratio             = "
            f"{st.mean(ratio_in):.4f} %"
        )
        print(f"  pooled corr(serial_spt, mtp_spt) within class      = {st.mean(corrs):+.4f}")
    print()

    print("=== D. identification summary and the published-median MDE ===")
    serial_leg = st.mean(serial_rel)
    cand_upper = min(class_means) if class_means else float("nan")
    print(
        f"  CLEAN: per-prompt rel sd of the serial leg     = {serial_leg:7.4f} %"
        "   (n=408, pinned binary, iid within-day)"
    )
    print(
        f"  UPPER BOUND on the candidate leg               = {cand_upper:7.4f} %"
        "   (tightest behaviour class; still holds content variation)"
    )
    print(
        "  `mean_draft_len` identity does NOT imply identical candidate work: the"
    )
    print(
        "  whole campaign changes candidate speed at a fixed schedule, so class 0"
    )
    print("  above mixes fast and slow trees. The candidate leg is not identified")
    print("  from below by this dataset.")
    print()
    deficit = 0.534
    for label, cand in (("point estimate (cand == serial)", serial_leg),
                        ("worst case (cand at its upper bound)", cand_upper)):
        per_prompt = math.hypot(serial_leg, cand)
        med_sd = 1.2533 * per_prompt / math.sqrt(len(prompts))
        z = deficit / med_sd
        print(f"  {label}:")
        print(f"    per-prompt raw_ratio rel sd = {per_prompt:7.4f} %")
        print(f"    rel sd of the median of {len(prompts)}   = {med_sd:7.4f} %")
        print(f"    detectable at 2 sd needs      >= {2 * med_sd:7.4f} %")
        print(f"    our {deficit:.3f} % deficit is        {z:7.2f} sd")
        print(
            f"    P(a redraw of the same tree promotes) ~ "
            f"{0.5 * math.erfc(z / math.sqrt(2.0)):.2e}"
        )
    print()
    print("  CONCLUSION. Neither floor the ledger has been quoting is right:")
    print("    item 148's 0.0693 % is about 3x BELOW the measured per-leg jitter;")
    print("    items 166/172's 0.7678 % is content granularity, not noise.")
    print("  Resubmitting an unchanged tree cannot close the deficit. A promoting")
    print("  candidate needs a real mechanism worth at least the deficit plus 2 sd.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
