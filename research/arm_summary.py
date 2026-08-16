#!/usr/bin/env python3
"""Side-by-side arm metrics and the schedule-invariant overhead decomposition."""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics

KEYS = [
    "all_tokens_matched",
    "residual_divergence_count",
    "declared_rows_total",
    "reference_checked_row_total",
    "round_count",
    "accepted_draft_rate",
    "accepted_draft_total",
    "rejected_draft_total",
    "effective_mean_draft_len",
    "verify_block_replayed_round_count",
    "seed_token_count",
    "target_cache_offset_final",
    "decode_seconds",
    "first_block_seconds",
    "max_block_request_seconds_after_first",
    "p50_block_request_seconds_after_first",
    "non_drafting_round_count",
    "max_rejected_tail_logit_delta",
    "parity_all_ok",
    "uses_pinned_mtp_head",
    "public_drift_tripwire_passed",
    "parent_measured_seconds_per_token",
]

ROUND_RE = re.compile(
    r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+).*?streak_in=(\d+) cap=(\d+)"
)


def load(runs):
    out = {}
    for lab, dr in runs.items():
        fs = sorted(glob.glob(dr + "/04-mtp-timed.json"))
        if not fs:
            continue
        out[lab] = json.load(open(fs[0]))
    return out


def position_acceptance(trace):
    """Conditional and unconditional realised acceptance per draft position.

    A speculative round evaluates all d drafts but keeps only the longest
    correct prefix, so the rate that drives the depth model is the conditional
    one: P(accept position i | positions 0..i-1 all accepted).
    """
    c_hits, c_obs = [0] * 12, [0] * 12
    u_hits, u_obs = [0] * 12, [0] * 12
    for line in open(trace, errors="ignore"):
        m = ROUND_RE.search(line)
        if not m:
            continue
        d, acc = int(m.group(2)), int(m.group(3))
        for i in range(d):
            u_obs[i] += 1
            if i < acc:
                u_hits[i] += 1
            if i <= acc:
                c_obs[i] += 1
                if i < acc:
                    c_hits[i] += 1
    return c_hits, c_obs, u_hits, u_obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/arm-summary.json")
    args = ap.parse_args()

    runs = {
        "I": "research/capture-runI",
        "J": "research/capture-runJ-gate2-512",
        "K": "research/capture-runK-gate2-cap8-512",
        "L": "research/capture-runL-gate1-cap8-512",
        "M": "research/capture-runM-gate0-cap8-512",
        "N": "research/capture-runN-gate1-cap8-512-confirm",
    }
    traces = {
        "I": "research/trace-runI-base-cap8-512.log",
        "J": "research/trace-runJ-gate2-512.log",
        "K": "research/trace-runK-gate2-cap8-512.log",
        "L": "research/trace-runL-gate1-cap8-512.log",
        "M": "research/trace-runM-gate0-cap8-512.log",
        "N": "research/trace-runN-gate1-cap8-512-confirm.log",
    }
    scores = {
        "I": "research/score-runI-base-cap8-512.json",
        "J": "research/score-runJ-gate2-512.json",
        "K": "research/score-runK-gate2-cap8-512.json",
        "L": "research/score-runL-gate1-cap8-512.json",
        "M": "research/score-runM-gate0-cap8-512.json",
        "N": "research/score-runN-gate1-cap8-512-confirm.json",
    }
    data = load(runs)
    labs = [x for x in ("I", "J", "K", "L", "M", "N") if x in data]

    print("=== headline ===")
    for lab in labs:
        s = json.load(open(scores[lab]))
        print("  Run %s local_ratio=%.10f passed=%s" % (lab, s["score"], s["passed"]))
    print()

    print("=== capture fields ===")
    for k in KEYS:
        row = "  %-40s" % k
        for lab in labs:
            row += " %s=%-20s" % (lab, str(data[lab].get(k))[:20])
        print(row)
    print()

    print("=== overhead decomposition ===")
    pts = []
    for lab in labs:
        bl = data[lab].get("block_request_seconds") or []
        tot = sum(bl)
        gap = data[lab]["decode_seconds"] - tot
        pts.append((len(bl), gap))
        print(
            "  Run %s rounds=%3d decode=%9.4f s blocks=%9.4f s gap=%8.4f s "
            "gap_per_token=%.4f ms" % (lab, len(bl), data[lab]["decode_seconds"], tot, gap, gap / 512 * 1000)
        )
    uniq = sorted({r for r, _ in pts})
    if len(uniq) >= 2:
        lo = min(pts, key=lambda p: p[0])
        hi = max(pts, key=lambda p: p[0])
        per_round = (hi[1] - lo[1]) / (hi[0] - lo[0])
        fixed = lo[1] - per_round * lo[0]
        print()
        print("  two-point fit over rounds %d..%d:" % (lo[0], hi[0]))
        print("    per_round_overhead = %.4f ms" % (per_round * 1000))
        print("    fixed_overhead     = %.4f s (%.4f ms/token)" % (fixed, fixed / 512 * 1000))
    print()

    print("=== realised per-position acceptance ===")
    posacc = {}
    for lab in labs:
        ch, co, uh, uo = position_acceptance(traces[lab])
        posacc[lab] = {
            "conditional_hits": ch,
            "conditional_obs": co,
            "unconditional_hits": uh,
            "unconditional_obs": uo,
        }
        print("  Run %s" % lab)
        print("    pos       " + "".join("%7d" % i for i in range(9)))
        print("    reached   " + "".join("%7d" % co[i] for i in range(9)))
        print(
            "    cond_acc  "
            + "".join(
                "%7s" % ("%.3f" % (ch[i] / co[i]) if co[i] else "-") for i in range(9)
            )
        )
        print(
            "    uncond    "
            + "".join(
                "%7s" % ("%.3f" % (uh[i] / uo[i]) if uo[i] else "-") for i in range(9)
            )
        )

    payload = {
        "headline": {lab: json.load(open(scores[lab]))["score"] for lab in labs},
        "capture": {lab: {k: data[lab].get(k) for k in KEYS} for lab in labs},
        "position_acceptance": posacc,
    }
    json.dump(payload, open(args.out, "w"), indent=2)
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
