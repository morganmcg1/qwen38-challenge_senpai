#!/usr/bin/env python3
"""Compare two official receipts prompt by prompt on the Yukon board.

The published score carries the runner's fresh serial draw at full weight, so a
published delta cannot separate a mechanism change from the lottery. Candidate
seconds per token come from the submitted workspace alone, so a per-prompt
candidate comparison between two runs is the mechanism signal.

    YUKON_API_TOKEN=... python3 research/board_per_prompt.py fetch
    python3 research/e87_s13_receipt_diff.py <idA> <idB> [<idC> ...]

With three or more ids from one unchanged tree the spread is a direct estimate
of the candidate-time replicate floor.
"""

import json
import statistics as st
import sys

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


def load():
    payload = json.load(open(CACHE))
    rows = payload
    for key in ("submissions", "rows", "data", "items"):
        if isinstance(rows, dict) and key in rows:
            rows = rows[key]
            break
    return [r for r in rows if isinstance(r, dict)]


def per_prompt(row):
    metrics = row.get("officialMetrics") or {}
    entries = metrics.get("per_prompt") if isinstance(metrics, dict) else None
    out = {}
    for entry in entries or []:
        sha = entry.get("prompt_sha256") or entry.get("promptSha256")
        if sha:
            out[sha[:8]] = entry
    return out


def field(entry, *names):
    for name in names:
        if name in entry:
            return entry[name]
    return None


def main(argv):
    rows = load()
    picked = []
    for want in argv:
        hit = [r for r in rows if str(r.get("id", "")).startswith(want)]
        if not hit:
            raise SystemExit("no submission starts with %s" % want)
        picked.append(hit[0])

    tables = [per_prompt(r) for r in picked]
    order = sorted(
        PROMPT_NAMES,
        key=lambda k: tables[0][k] and field(tables[0][k], "mtp_seconds_per_token_mean", "candidate_mtp_seconds_per_token_mean") or 0,
    )

    head = "  %-9s" % "prompt"
    for r in picked:
        head += " %14s" % r["id"][:8]
    head += " %10s %10s %9s" % ("delta %", "draft len", "nondraft")
    print(head)

    deltas = []
    for key in order:
        name = PROMPT_NAMES[key]
        vals, drafts, nond = [], [], []
        for table in tables:
            entry = table.get(key)
            if entry is None:
                vals.append(float("nan"))
                continue
            vals.append(
                field(entry, "mtp_seconds_per_token_mean",
                      "candidate_mtp_seconds_per_token_mean"))
            drafts.append(field(entry, "effective_mean_draft_len",
                                "effectiveMeanDraftLen"))
            nond.append(field(entry, "non_drafting_round_count",
                              "non_drafting_rounds"))
        pct = 100.0 * (vals[-1] - vals[0]) / vals[0]
        deltas.append(pct)
        line = "  %-9s" % name
        for v in vals:
            line += " %14.8f" % v
        same_draft = len(set("%.6f" % d for d in drafts if d is not None)) <= 1
        line += " %+9.3f %10s %9s" % (
            pct,
            ("%.3f" % drafts[0]) + ("" if same_draft else " DIFFER"),
            nond[0] if nond else "?",
        )
        print(line)

    print()
    print("  mean candidate delta   %+.3f %%" % st.mean(deltas))
    print("  median candidate delta %+.3f %%" % st.median(deltas))
    if len(deltas) > 1:
        print("  spread of the delta    %.3f %% sd" % st.pstdev(deltas))
    for r in picked:
        print("  %s  official %.8f  %s  commit %s" % (
            r["id"][:8], r.get("officialScore") or float("nan"),
            r.get("status"), str(r.get("submissionCommitSha"))[:7]))

    widths = [1.0 + field(tables[0][k], "effective_mean_draft_len") for k in order]
    shape_report(widths, deltas, order)
    mode_report(rows, picked, tables)


def shape_report(widths, deltas, order):
    """Separate a fixed per-call penalty from a per-row cost.

    A cost paid per verified row rises with verify width.  A cost paid once per
    target call does not, and once expressed as a percentage of a round whose
    time grows with width, it falls.  That contrast identified the `b3f88ed2`
    Q-row rider.  It needs no round-count reconstruction, because two runs that
    share a schedule share round counts prompt by prompt, so the ratio of round
    times equals the ratio of per-token times.

    `plutarch` is excluded from the fit.  It is about 92 % non-drafting, so it
    barely exercises the drafting path at all and is the control rather than a
    point on the curve.
    """
    pts = [(w, d, PROMPT_NAMES[k]) for w, d, k in zip(widths, deltas, order)
           if PROMPT_NAMES[k] != "plutarch"]
    n = len(pts)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx, my = st.fmean(xs), st.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    inter = my - slope * mx
    resid = [y - inter - slope * x for x, y in zip(xs, ys)]
    se = ((sum(r * r for r in resid) / (n - 2)) / sxx) ** 0.5
    print()
    print("  delta %% against verify width (1 + draft len), %d drafting prompts,"
          " plutarch excluded" % n)
    print("    slope     %+.4f %% per row   se %.4f   t %+.2f" % (slope, se, slope / se))
    print("    intercept %+.4f %%   at width 1" % inter)
    if abs(slope / se) < 2.0:
        verdict = "no width dependence resolved"
    elif slope > 0:
        verdict = "RISING: a per-row cost, it scales with verify width"
    else:
        verdict = "FALLING: a fixed per-call cost diluted by wider rounds"
    print("    %s" % verdict)
    plut = [(w, d) for w, d, k in zip(widths, deltas, order)
            if PROMPT_NAMES[k] == "plutarch"]
    if plut:
        print("    control  plutarch width %.3f  delta %+.3f %%  "
              "(92 %% non-drafting: near mode-free)" % (plut[0][0], plut[0][1]))


def mode_report(rows, picked, tables):
    """Locate each run in the measurement-speed distribution of its own cohort.

    The advisor's `L / plutarch` diagnostic needs per-prompt round counts.  The
    board publishes `effective_mean_draft_len` and `non_drafting_round_count`
    but not accepted counts, and the scheduler picks depth adaptively, so round
    counts are not recoverable from public fields.  `plutarch` itself is the
    part of that diagnostic that is recoverable: it is about 92 % non-drafting,
    so its candidate time is a near mode-free probe of machine speed.
    """
    key = "c1ec5866"
    sched = tuple(
        (field(e, "effective_mean_draft_len"), field(e, "non_drafting_round_count"))
        for _, e in sorted(tables[-1].items())
    )
    cohort = []
    for r in rows:
        if r.get("officialScore") is None:
            continue
        t = per_prompt(r)
        if len(t) != 8 or key not in t:
            continue
        s = tuple(
            (field(e, "effective_mean_draft_len"), field(e, "non_drafting_round_count"))
            for _, e in sorted(t.items())
        )
        if s == sched:
            cohort.append((field(t[key], "mtp_seconds_per_token_mean"), r["id"][:8]))
    cohort.sort()
    print()
    print("  measurement mode: plutarch candidate seconds per token, "
          "same-schedule cohort of %d" % len(cohort))
    if not cohort:
        return
    times = [c[0] for c in cohort]
    print("    cohort min %.8f  median %.8f  max %.8f"
          % (times[0], st.median(times), times[-1]))
    for r in picked:
        rid = r["id"][:8]
        hit = [i for i, c in enumerate(cohort) if c[1] == rid]
        if not hit:
            print("    %s  not in this cohort" % rid)
            continue
        i = hit[0]
        print("    %s  %.8f  rank %d of %d  (%.0f th pct, lower is faster)"
              % (rid, cohort[i][0], i + 1, len(cohort),
                 100.0 * i / max(len(cohort) - 1, 1)))


if __name__ == "__main__":
    main(sys.argv[1:])
