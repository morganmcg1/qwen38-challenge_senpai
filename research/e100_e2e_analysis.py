#!/usr/bin/env python3
"""E100: reduce the end-to-end ABBA sessions to a decode-time effect.

Three sessions run on the same two builds, each counterbalanced across its own
four slots:

  d8    64 decode tokens, offered depth 8. The shipped schedule. Measures reach
        and conversion together.
  d4    64 decode tokens, offered depth 4. The reach control: the offered
        ceiling caps draftCount at 4, so M = 5 dominates and this isolates
        conversion alone.
  w512  512 decode tokens, offered depth 8. The ranked decode window. The timed
        leg always carries the same 512-token seed prefill, so at 64 decode
        tokens the prefill is most of the leg and divides any decode-side
        effect by about six before it reaches seconds_per_token. This session
        removes most of that dilution and multiplies the round count by eight.

Arm means are (A1 + A2) / 2 and (B1 + B2) / 2, which cancels linear thermal
drift to first order across the counterbalanced order.

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

SLOTS = (("a1", "collapse"), ("b1", "base"), ("b2", "base"), ("a2", "collapse"))

# name, tag infix, offered depth, decode tokens.
#
# Every session runs the M = 5 only arm: `<T,5,5,true>` with `<T,9,5,true>`
# dropped, so `worker_m9_ipg5` is 0 in both arms.
SESSIONS = (
    ("d8", "d8", 8, 64),
    ("w512", "w512", 8, 512),
    ("w512d4", "w512d4", 4, 512),
)


def load(infix, slot):
    tag = "e100-e2e-%s-%s" % (infix, slot)
    root = os.path.join("research/out", tag)
    if not os.path.isdir(root):
        return None
    with open(os.path.join(root, "score.json")) as f:
        score = json.load(f)["metrics"]
    meta = {}
    with open(os.path.join(root, "meta.txt")) as f:
        for line in f:
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                meta[k] = v
    return dict(tag=tag, score=score, meta=meta)


def check(leg, arm, depth, tokens):
    m, s = leg["meta"], leg["score"]
    want5 = "1" if arm == "collapse" else "0"
    problems = []
    # The binary is the only witness that decides which kernel ran. The twin
    # fields below only prove the source was edited; see FINDING 28.
    for field, want in (("worker_m5_ipg5", want5),
                        ("worker_m5_ipg3", "0" if arm == "collapse" else "1"),
                        ("worker_m6_ipg3", "1"),
                        ("worker_m6_ipg2", "0")):
        if m.get(field) != want:
            problems.append("%s=%s expected %s" % (field, m.get(field), want))
    if m.get("worker_sha256_pre") != m.get("worker_sha256_post"):
        problems.append("worker changed during the leg")
    if m.get("twin_m5") != want5:
        problems.append("twin_m5=%s expected %s" % (m.get("twin_m5"), want5))
    if m.get("twin_m9") != "0":
        problems.append("twin_m9=%s expected 0" % m.get("twin_m9"))
    if m.get("arm") != arm:
        problems.append("arm=%s expected %s" % (m.get("arm"), arm))
    if m.get("offered_depth") != str(depth):
        problems.append("offered_depth=%s expected %d"
                        % (m.get("offered_depth"), depth))
    if int(m.get("decode_tokens", "64")) != tokens:
        problems.append("decode_tokens=%s expected %d"
                        % (m.get("decode_tokens"), tokens))
    if s.get("decode_tokens") != tokens:
        problems.append("score decode_tokens=%s expected %d"
                        % (s.get("decode_tokens"), tokens))
    if not s["all_tokens_matched"]:
        problems.append("all_tokens_matched=false")
    if s["residual_divergence_count"]:
        problems.append("residual_divergence_count=%d"
                        % s["residual_divergence_count"])
    if m.get("git_dirty_build", "0") != "0":
        problems.append("git_dirty_build=%s" % m.get("git_dirty_build"))
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", dest="out")
    args = ap.parse_args()

    legs, sessions = {}, []
    for name, infix, depth, tokens in SESSIONS:
        present = {}
        for slot, arm in SLOTS:
            leg = load(infix, slot)
            if leg is None:
                continue
            problems = check(leg, arm, depth, tokens)
            if problems:
                print("leg %s: %s" % (leg["tag"], "; ".join(problems)))
                return 2
            present[slot] = leg
            legs[leg["tag"]] = leg
        if present:
            sessions.append((name, depth, tokens, present))

    if not sessions:
        print("no legs found under research/out")
        return 2

    print("=" * 100)
    print("IDENTITY")
    print("=" * 100)
    for name, _, _, present in sessions:
        for slot, _ in SLOTS:
            leg = present.get(slot)
            if leg is None:
                continue
            m, s = leg["meta"], leg["score"]
            print("  %-20s arm=%-9s head=%s dirty=%s entry_c=%5.1f "
                  "exit_c=%5.1f matched=%s divergences=%d"
                  % (leg["tag"], m["arm"], m["git_head"][:8], m["git_dirty"],
                     float(m["gpu_temp_entry_c"]), float(m["gpu_temp_exit_c"]),
                     s["all_tokens_matched"], s["residual_divergence_count"]))
    for tag, leg in sorted(legs.items()):
        if leg["meta"].get("git_dirty") != "0":
            print("  ADVISORY %s ran with %s uncommitted path(s); no build "
                  "input was dirty (git_dirty_build=%s)"
                  % (tag, leg["meta"]["git_dirty"],
                     leg["meta"].get("git_dirty_build", "not recorded")))
    temps = [float(l["meta"]["gpu_temp_entry_c"]) for l in legs.values()]
    print("  entry temperature spread: %.1f C (min %.1f, max %.1f)"
          % (max(temps) - min(temps), min(temps), max(temps)))
    print("  cool_gate_passed_real_gate=false  gate_qualified_for_timing=false"
          "  timing_valid=false")

    summary = {}
    for name, depth, tokens, present in sessions:
        print()
        print("=" * 100)
        print("SESSION %s -- %d decode tokens, offered depth %d, %d/4 slots"
              % (name, tokens, depth, len(present)))
        print("=" * 100)
        print("  %-20s %11s %12s %9s %11s %8s"
              % ("leg", "mtp_s/tok", "serial_s/tok", "ratio", "mean_draft",
                 "accept"))
        for slot, _ in SLOTS:
            leg = present.get(slot)
            if leg is None:
                continue
            s = leg["score"]
            print("  %-20s %11.6f %12.6f %9.4f %11.4f %8.3f"
                  % (leg["tag"], s["mtp_seconds_per_token"],
                     s["serial_seconds_per_token"], s["mtp_decode_speedup"],
                     s["effective_mean_draft_len"], s["accepted_draft_rate"]))

        base_slots = [s for s in ("b1", "b2") if s in present]
        coll_slots = [s for s in ("a1", "a2") if s in present]
        out = {"decode_tokens": tokens, "offered_depth": depth,
               "base_slots": base_slots, "collapse_slots": coll_slots}
        for key, label, better in (
            ("mtp_seconds_per_token", "candidate s/tok", "lower"),
            ("serial_seconds_per_token", "serial s/tok", "lower"),
            ("mtp_decode_speedup", "local ratio", "higher"),
            ("effective_mean_draft_len", "mean draft len", "-"),
        ):
            def mean(slots):
                v = [present[s]["score"][key] for s in slots]
                return sum(v) / len(v)

            def spread(slots):
                v = [present[s]["score"][key] for s in slots]
                return (max(v) - min(v)) if len(v) > 1 else float("nan")

            b, c = mean(base_slots), mean(coll_slots)
            delta = 100.0 * (c / b - 1.0)
            out[key] = dict(base=b, collapse=c, delta_pct=delta,
                            base_spread_pct=100.0 * spread(base_slots) / b,
                            collapse_spread_pct=100.0 * spread(coll_slots) / b)
            print("  %-20s base %11.6f  collapse %11.6f  delta %+7.3f %% "
                  "(%s is better; within-arm spread base %.3f %% "
                  "collapse %.3f %%)"
                  % (label, b, c, delta, better,
                     out[key]["base_spread_pct"],
                     out[key]["collapse_spread_pct"]))
        summary[name] = out

    print()
    print("=" * 100)
    print("READING")
    print("=" * 100)
    for name, _, tokens, _ in sessions:
        s = summary[name]["mtp_seconds_per_token"]
        spreads = [x for x in (s["base_spread_pct"], s["collapse_spread_pct"])
                   if x == x]
        worst = max(spreads) if spreads else float("nan")
        print("  %-6s (%3d tok, depth %d): candidate s/tok %+7.3f %%   "
              "worst within-arm spread %.3f %%"
              % (name, tokens, summary[name]["offered_depth"],
                 s["delta_pct"], worst))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(
                summary=summary,
                legs={t: dict(meta=l["meta"], score=l["score"])
                      for t, l in legs.items()},
            ), f, indent=2, sort_keys=True)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
