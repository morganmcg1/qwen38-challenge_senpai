#!/usr/bin/env python3
"""E100: reduce the end-to-end ABBA session to a decode-time effect per depth.

Two depth sessions run on the same four builds:

  depth 8   the shipped schedule. `segmentedVerifyDepthCap` is 7, so the verify
            width is 1..8 and M = 5 is one width among several. This measures
            reach and conversion together, which is what a ranked run sees.
  depth 4   the reach control. The offered ceiling caps draftCount at 4, so the
            verify width is at most 5 and M = 5 dominates. This measures
            conversion alone.

Arm means are (A1 + A2) / 2 and (B1 + B2) / 2, which cancels linear thermal
drift to first order across the A B B A order.

The headline is ABSOLUTE candidate seconds per token, not the local ratio. Both
local legs run the same build, so a change that speeds the target generally
cancels in the ratio; a dispatch change that only fires on the drafting leg does
not. Both are reported.

Usage:
  research/e100_e2e_analysis.py [--json OUT]
"""

import argparse
import json
import os
import sys

SLOTS = (("a1", "base"), ("b1", "collapse"), ("b2", "collapse"), ("a2", "base"))
DEPTHS = (8, 4)


def load(depth, slot):
    tag = "e100-e2e-d%d-%s" % (depth, slot)
    root = os.path.join("research/out", tag)
    with open(os.path.join(root, "score.json")) as f:
        score = json.load(f)["metrics"]
    meta = {}
    with open(os.path.join(root, "meta.txt")) as f:
        for line in f:
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                meta[k] = v
    return dict(tag=tag, score=score, meta=meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", dest="out")
    args = ap.parse_args()

    legs = {}
    for depth in DEPTHS:
        for slot, arm in SLOTS:
            leg = load(depth, slot)
            want5 = "1" if arm == "collapse" else "0"
            if leg["meta"].get("twin_m5") != want5:
                print("leg %s witness says twin_m5=%s, expected %s for arm %s"
                      % (leg["tag"], leg["meta"].get("twin_m5"), want5, arm))
                return 2
            if leg["meta"].get("arm") != arm:
                print("leg %s declares arm=%s" % (leg["tag"], leg["meta"]["arm"]))
                return 2
            if leg["meta"].get("offered_depth") != str(depth):
                print("leg %s declares offered_depth=%s"
                      % (leg["tag"], leg["meta"].get("offered_depth")))
                return 2
            if not leg["score"]["all_tokens_matched"]:
                print("leg %s did not match all tokens" % leg["tag"])
                return 2
            legs[(depth, slot)] = leg

    print("=" * 96)
    print("IDENTITY")
    print("=" * 96)
    for depth in DEPTHS:
        for slot, _ in SLOTS:
            leg = legs[(depth, slot)]
            m, s = leg["meta"], leg["score"]
            print("  %-18s arm=%-9s head=%s dirty=%s entry_c=%5.1f exit_c=%5.1f "
                  "matched=%s divergences=%d"
                  % (leg["tag"], m["arm"], m["git_head"][:8], m["git_dirty"],
                     float(m["gpu_temp_entry_c"]), float(m["gpu_temp_exit_c"]),
                     s["all_tokens_matched"], s["residual_divergence_count"]))
    temps = [float(l["meta"]["gpu_temp_entry_c"]) for l in legs.values()]
    print("  entry temperature spread: %.1f C (min %.1f, max %.1f)"
          % (max(temps) - min(temps), min(temps), max(temps)))
    print("  cool_gate_passed_real_gate=false  gate_qualified_for_timing=false")

    summary = {}
    for depth in DEPTHS:
        print()
        print("=" * 96)
        print("OFFERED DEPTH %d" % depth)
        print("=" * 96)
        rows = []
        print("  %-18s %11s %11s %9s %11s %8s"
              % ("leg", "mtp_s/tok", "serial_s/tok", "ratio", "mean_draft",
                 "accept"))
        for slot, arm in SLOTS:
            s = legs[(depth, slot)]["score"]
            rows.append(s)
            print("  %-18s %11.6f %11.6f %9.4f %11.4f %8.3f"
                  % (legs[(depth, slot)]["tag"], s["mtp_seconds_per_token"],
                     s["serial_seconds_per_token"], s["mtp_decode_speedup"],
                     s["effective_mean_draft_len"], s["accepted_draft_rate"]))

        def arm_mean(key, slots):
            v = [legs[(depth, s)]["score"][key] for s in slots]
            return sum(v) / len(v)

        base_slots, coll_slots = ("a1", "a2"), ("b1", "b2")
        out = {}
        for key, label, better in (
            ("mtp_seconds_per_token", "candidate s/tok", "lower"),
            ("serial_seconds_per_token", "serial s/tok", "lower"),
            ("mtp_decode_speedup", "local ratio", "higher"),
            ("effective_mean_draft_len", "mean draft len", "-"),
        ):
            b, c = arm_mean(key, base_slots), arm_mean(key, coll_slots)
            delta = 100.0 * (c / b - 1.0)
            spread = 100.0 * abs(
                legs[(depth, "a1")]["score"][key]
                - legs[(depth, "a2")]["score"][key]) / b
            out[key] = dict(base=b, collapse=c, delta_pct=delta,
                            base_a1_a2_spread_pct=spread)
            print("  %-18s base %11.6f  collapse %11.6f  delta %+7.3f %% "
                  "(%s is better; A1-A2 spread %.3f %%)"
                  % (label, b, c, delta, better, spread))
        summary["depth_%d" % depth] = out

    print()
    print("=" * 96)
    print("READING")
    print("=" * 96)
    d4 = summary["depth_4"]["mtp_seconds_per_token"]
    d8 = summary["depth_8"]["mtp_seconds_per_token"]
    print("  conversion (depth 4, M = 5 dominant): %+7.3f %%" % d4["delta_pct"])
    print("  shipped schedule (depth 8):           %+7.3f %%" % d8["delta_pct"])
    print("  A1-to-A2 repeatability, worst of the two sessions: %.3f %%"
          % max(d4["base_a1_a2_spread_pct"], d8["base_a1_a2_spread_pct"]))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(
                summary=summary,
                legs={l["tag"]: dict(meta=l["meta"], score=l["score"])
                      for l in legs.values()},
            ), f, indent=2, sort_keys=True)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
