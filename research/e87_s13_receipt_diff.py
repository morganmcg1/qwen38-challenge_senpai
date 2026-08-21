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


if __name__ == "__main__":
    main(sys.argv[1:])
