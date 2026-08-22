#!/usr/bin/env python3
"""Audit the F76 mode index against the whole Yukon board.

    usage: research/e126_modeaudit.py [--out PATH]

The index is a zero-sum weighting of the eight per-prompt candidate decode
times, so a uniform speedup cancels and only a non-uniform one moves it. The
campaign reads a move of about 1.000 units as a change of hidden draw mode
(Rule 63).

This audit asks whether that reading is identifiable. Three checks:

1. Is the index bimodal over every scored row on the board?
2. Does the same submitted commit, re-run, land in a different mode? A genuine
   draw must vary when the tree is held fixed.
3. Does the serial leg carry the same structure? The serial numerator comes
   from the runner's own prebuilt baseline workspace and candidate-editable
   code cannot change it, so a property of the run rather than of the tree has
   to appear there too.

Check 3 is the discriminator. `research/e126_modeindex.py` holds the weights
and the decision layer; this file only audits them.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import statistics as st

from e126_modeindex import PROMPT_NAMES, SAME_MODE_SD, THRESHOLD, WEIGHTS

BOARD = pathlib.Path("/tmp/yukon-board/full.json")

# The prompt that barely speculates. Its weight is the second largest in the
# index, so any mechanism that acts only where drafting happens moves the
# index without any change of draw.
NON_DRAFTING = "plutarch"


def scored():
    if not BOARD.exists():
        raise SystemExit("run `python3 research/board_per_prompt.py fetch` first")
    for row in json.loads(BOARD.read_text()):
        per_prompt = {}
        for entry in (row.get("officialMetrics") or {}).get("per_prompt") or []:
            name = PROMPT_NAMES.get(str(entry.get("prompt_sha256", ""))[:8])
            if name:
                per_prompt[name] = entry
        if len(per_prompt) == 8 and row.get("officialScore"):
            yield row, per_prompt


def weighted(per_prompt, field):
    return sum(w * 100.0 * math.log(per_prompt[p][field])
               for p, w in WEIGHTS.items())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    rows = list(scored())
    report: dict = {"harness": "ranked", "scored_rows": len(rows)}
    print("harness=ranked. %d scored rows with all eight prompts" % len(rows))

    mtp = [weighted(p, "mtp_seconds_per_token_mean") for _, p in rows]
    serial = [weighted(p, "serial_seconds_per_token_mean") for _, p in rows]

    print("\n=== 1. is the candidate-leg index bimodal? ===")
    lo, width, nbins = -14.2, 0.1, 24
    hist = [0] * nbins
    for v in mtp:
        k = int((v - lo) / width)
        if 0 <= k < nbins:
            hist[k] += 1
    for i, count in enumerate(hist):
        print("  %+6.1f %4d %s" % (lo + i * width, count, "#" * min(count, 60)))
    inside = [v for v in mtp if lo <= v < lo + nbins * width]
    print("  %d of %d rows inside the window" % (len(inside), len(mtp)))
    report["index_window_rows"] = len(inside)
    report["index_histogram"] = {"lo": lo, "width": width, "counts": hist}

    print("\n=== 2. does one commit, re-run, change mode? ===")
    by_commit = collections.defaultdict(list)
    for (row, _), index in zip(rows, mtp):
        sha = row.get("submissionCommitSha")
        if sha:
            by_commit[sha].append((row["id"][:8], row.get("solverUsername"),
                                   float(row["officialScore"]), index))
    repeats = {k: v for k, v in by_commit.items() if len(v) > 1}
    spreads = []
    for sha, seen in sorted(repeats.items(), key=lambda kv: -len(kv[1])):
        spread = max(x[3] for x in seen) - min(x[3] for x in seen)
        spreads.append(spread)
        if spread > 0.5:
            print("  %s n=%d index spread %.4f" % (sha[:10], len(seen), spread))
            for entry in sorted(seen, key=lambda y: y[3]):
                print("     %-9s %-12s score %.6f index %+8.4f" % entry)
    print("  %d commits submitted more than once" % len(repeats))
    if spreads:
        print("  index spread within a fixed commit: median %.4f, max %.4f, "
              "%d of %d exceed half a mode flip"
              % (st.median(spreads), max(spreads),
                 sum(1 for s in spreads if s > 0.5), len(spreads)))
    report["repeat_commits"] = len(repeats)
    report["repeat_index_spread_median"] = st.median(spreads) if spreads else None
    report["repeat_index_spread_max"] = max(spreads) if spreads else None
    report["repeat_index_spread_over_half_flip"] = sum(
        1 for s in spreads if s > 0.5)

    print("\n=== 3. does the serial leg carry the same structure? ===")
    print("  candidate-leg index: sd %.4f, range %.3f" %
          (st.pstdev(mtp), max(mtp) - min(mtp)))
    print("  serial-leg    index: sd %.4f, range %.3f" %
          (st.pstdev(serial), max(serial) - min(serial)))
    total = [sum(p[n]["serial_seconds_per_token_mean"] for n in WEIGHTS)
             for _, p in rows]
    print("  serial sum of the eight prompts: %.5f .. %.5f, spread %.2f %%"
          % (min(total), max(total), 100.0 * (max(total) / min(total) - 1.0)))
    report["mtp_index_sd"] = st.pstdev(mtp)
    report["serial_index_sd"] = st.pstdev(serial)
    report["serial_sum_spread_pct"] = 100.0 * (max(total) / min(total) - 1.0)

    print("\n=== 4. sensitivity to a drafting-only mechanism ===")
    leak = sum(w for p, w in WEIGHTS.items() if p != NON_DRAFTING)
    print("  weights over the seven drafting prompts sum to %+.4f" % leak)
    print("  a 1 %% speedup confined to them moves the index %+.4f units,"
          % -leak)
    print("  which is %.1f same-mode sd and %.0f %% of one mode flip."
          % (abs(leak) / SAME_MODE_SD, 100.0 * abs(leak)))
    print("  %s carries w %+.4f and is the only prompt that barely drafts."
          % (NON_DRAFTING, WEIGHTS[NON_DRAFTING]))
    report["drafting_weight_sum"] = leak
    report["pct_per_1pct_drafting_speedup"] = -leak
    report["threshold"] = THRESHOLD

    print("\n=== 5. the anchors, in both legs ===")
    from e126_modeindex import ANCHORS

    watch = dict(ANCHORS)
    watch["cf9a9eda"] = ("morganmcg1", 3.26815344, None, "rejected")
    print("  %-9s %-12s %11s %9s %9s %9s"
          % ("id", "solver", "published", "mtp idx", "serial", "draft_p"))
    anchor_rows = []
    for key in watch:
        hit = [(r, p) for r, p in rows if str(r["id"]).startswith(key[:8])]
        if not hit:
            continue
        row, per_prompt = hit[0]
        mi = weighted(per_prompt, "mtp_seconds_per_token_mean")
        si = weighted(per_prompt, "serial_seconds_per_token_mean")
        anchor_rows.append((key, row.get("solverUsername"),
                            float(row["officialScore"]), mi, si,
                            per_prompt[NON_DRAFTING]["effective_mean_draft_len"]))
    for entry in sorted(anchor_rows, key=lambda e: e[3]):
        print("  %-9s %-12s %11.8f %+9.4f %+9.4f %9.3f" % entry)
    if len(anchor_rows) > 2:
        mi = [e[3] for e in anchor_rows]
        si = [e[4] for e in anchor_rows]
        print("  over these anchors: mtp idx sd %.4f, serial idx sd %.4f"
              % (st.pstdev(mi), st.pstdev(si)))
        mbar, sbar = st.fmean(mi), st.fmean(si)
        num = sum((a - mbar) * (b - sbar) for a, b in zip(mi, si))
        den = math.sqrt(sum((a - mbar) ** 2 for a in mi)
                        * sum((b - sbar) ** 2 for b in si))
        print("  correlation between the two legs: r %+.3f" % (num / den))
        report["anchor_leg_correlation"] = num / den
        report["anchor_mtp_index_sd"] = st.pstdev(mi)
        report["anchor_serial_index_sd"] = st.pstdev(si)
    report["anchors"] = [
        {"id": e[0], "solver": e[1], "published": e[2], "mtp_index": e[3],
         "serial_index": e[4]} for e in anchor_rows]

    if args.out:
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
