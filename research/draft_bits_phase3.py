#!/usr/bin/env python3
"""Research-only: analyse a counterbalanced ABBA draft-bits timing experiment.

    research/draft_bits_phase3.py JOB_LOG --prefix TAG_PREFIX [--order 4,3,3,4]

Phase 2 compared one control run against one candidate run that started 7.96C
hotter, so its per-round delta and its thermal drift were confounded. Phase 3
runs each arm twice in an ABBA order and this script separates the two.

Three estimates are reported and they answer different questions:

  * per-pair deltas, one per adjacent (control, candidate) pair, whose spread
    bounds the residual after counterbalancing;
  * the ABBA-combined delta, which averages the replicates of each arm and so
    cancels any drift that is linear in position;
  * the control-vs-control delta between the two identical arm-4 replicates at
    positions 1 and 4, which measures this host's own reproducibility over the
    span of the experiment and is the noise floor the candidate must beat.

The per-round term and the acceptance term stay separate throughout, because a
precision change that buys bandwidth and one that changes which tokens were
drafted are different findings.
"""
import argparse
import json
import os
import statistics as st
import sys

from draft_bits_phase2 import FIELDS, parse


def replicate_dirs(root, prefix, order):
    return [(i + 1, b, os.path.join(root, "%s-p%d-b%s" % (prefix, i + 1, b)))
            for i, b in enumerate(order)]


def adjacent_pairs(order):
    """Pair each consecutive control/candidate neighbour, consuming both."""
    pairs, i = [], 0
    while i + 1 < len(order):
        if order[i] != order[i + 1]:
            pairs.append((i + 1, i + 2))
            i += 2
        else:
            i += 1
    return pairs


def round_delta(ctl, cand):
    n = min(len(ctl), len(cand))
    return [(ctl[i], cand[i]) for i in range(n) if ctl[i]["d"] == cand[i]["d"]]


def report_pair(label, ctl, cand):
    pairs = round_delta(ctl, cand)
    print("  %s: paired rounds=%d of %d" % (label, len(pairs), min(len(ctl), len(cand))))
    for f in FIELDS:
        d = [a[f] - b[f] for a, b in pairs]
        print("    %-16s median=%+8.1f us  mean=%+8.1f us"
              % (f, st.median(d), sum(d) / len(d)))
    steady = pairs[1:]
    dr = [a["round_us"] - b["round_us"] for a, b in steady]
    readouts = sum(a["d"] for a, _ in steady)
    per = sum(dr) / readouts if readouts else float("nan")
    print("    steady rounds=%d total_round_us=%+d readouts=%d per_readout=%+.1f us"
          % (len(steady), sum(dr), readouts, per))
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--order", default="4,3,3,4")
    ap.add_argument("--root", default=".mlxfast-private/draft-bits")
    ap.add_argument("--control", default="4")
    args = ap.parse_args()
    order = args.order.split(",")

    blocks = parse(args.log)
    timed = {}
    for b in blocks:
        if b["rounds"]:
            if b["slot"] in timed:
                sys.exit("slot %s has more than one timed instance" % b["slot"])
            timed[b["slot"]] = b["rounds"]
    if sorted(timed) != list(range(1, len(order) + 1)):
        sys.exit("expected timed rounds for slots 1..%d, got %s"
                 % (len(order), sorted(timed)))

    print("== draft-head provenance (every worker instance) ==")
    for b in blocks:
        print("  slot=%s bits=%s source_bits=%s requant_ms=%.3f rounds=%d"
              % (b["slot"], b["bits"], b["source_bits"], b["requant_ms"],
                 len(b["rounds"])))

    print("\n== acceptance term ==")
    sched = {}
    for slot, bits in enumerate(order, 1):
        r = timed[slot]
        key = tuple((x["d"], x["acc"]) for x in r)
        sched.setdefault(key, []).append("p%d-b%s" % (slot, bits))
        print("  slot=%d bits=%s rounds=%d proposed=%d accepted=%d emitted=%d"
              % (slot, bits, len(r), sum(x["d"] for x in r),
                 sum(x["acc"] for x in r), len(r) + sum(x["acc"] for x in r)))
    if len(sched) == 1:
        print("  all %d replicates share one (depth, accepted) schedule"
              " => acceptance term is exactly 0" % len(order))
    else:
        print("  WARNING: %d distinct schedules: %s"
              % (len(sched), list(sched.values())))

    print("\n== per-pair per-round delta (control - candidate) ==")
    per_pair = []
    for a, b in adjacent_pairs(order):
        ctl, cand = (a, b) if order[a - 1] == args.control else (b, a)
        per_pair.append(report_pair("pair p%d(b%s) - p%d(b%s)"
                                    % (ctl, order[ctl - 1], cand, order[cand - 1]),
                                    timed[ctl], timed[cand]))
    if len(per_pair) > 1:
        print("  per-readout across pairs: %s  spread=%.1f us"
              % (" ".join("%+.1f" % p for p in per_pair),
                 max(per_pair) - min(per_pair)))

    ctl_slots = [s for s, b in enumerate(order, 1) if b == args.control]
    cand_slots = [s for s, b in enumerate(order, 1) if b != args.control]
    print("\n== ABBA-combined delta (mean control slots %s - mean candidate slots %s) =="
          % (ctl_slots, cand_slots))
    n = min(len(timed[s]) for s in timed)
    depths = [timed[ctl_slots[0]][i]["d"] for i in range(n)]
    combined = []
    for i in range(n):
        if any(timed[s][i]["d"] != depths[i] for s in timed):
            continue
        row = {"d": depths[i]}
        for f in FIELDS:
            row[f] = (st.mean(timed[s][i][f] for s in ctl_slots)
                      - st.mean(timed[s][i][f] for s in cand_slots))
        combined.append(row)
    print("  paired rounds=%d of %d" % (len(combined), n))
    for f in FIELDS:
        d = [r[f] for r in combined]
        print("    %-16s median=%+8.1f us  mean=%+8.1f us"
              % (f, st.median(d), sum(d) / len(d)))
    steady = combined[1:]
    tot = sum(r["round_us"] for r in steady)
    readouts = sum(r["d"] for r in steady)
    print("    steady rounds=%d total_round_us=%+.0f readouts=%d per_readout=%+.1f us"
          % (len(steady), tot, readouts, tot / readouts))

    if len(ctl_slots) == 2:
        print("\n== control-vs-control noise floor (p%d vs p%d, identical arms) =="
              % tuple(ctl_slots))
        report_pair("ctl p%d - ctl p%d" % tuple(ctl_slots),
                    timed[ctl_slots[0]], timed[ctl_slots[1]])

    print("\n== legs, temperatures, and ABBA-combined headline ==")
    legs = {}
    for slot, bits, d in replicate_dirs(args.root, args.prefix, order):
        path = os.path.join(d, "amdahl.json")
        if not os.path.exists(path):
            print("  slot=%d bits=%s MISSING %s" % (slot, bits, path))
            continue
        r = json.load(open(path))
        ident = {}
        for line in open(os.path.join(d, "identity.txt")):
            for tok in line.replace("run-draft-bits-arm:", "").split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    ident[k] = v
        legs[slot] = {
            "bits": bits,
            "mtp": r["mtp_leg"]["parent_measured_seconds_per_token"],
            "serial": r["serial_leg"]["parent_measured_seconds_per_token"],
            "matched": (r["mtp_leg"]["all_tokens_matched"]
                        and r["serial_leg"]["all_tokens_matched"]),
            "score": r["amdahl"]["measured_local_score"],
            "before": float(ident.get("gpu_temp_c_before", "nan")),
            "after": float(ident.get("gpu_temp_c_after", "nan")),
        }
        print("  slot=%d bits=%s matched=%s mtp_spt=%.10f serial_spt=%.10f "
              "temp %.2f->%.2f C" % (slot, bits, legs[slot]["matched"],
                                     legs[slot]["mtp"], legs[slot]["serial"],
                                     legs[slot]["before"], legs[slot]["after"]))
    if len(legs) == len(order):
        for field in ("mtp", "serial"):
            c = st.mean(legs[s][field] for s in ctl_slots)
            k = st.mean(legs[s][field] for s in cand_slots)
            print("  %-6s control=%.10f candidate=%.10f delta=%+.4f%%"
                  % (field, c, k, 100.0 * (k - c) / c))
        a, b = ctl_slots
        print("  control-vs-control mtp drift p%d->p%d = %+.4f%%  (noise floor)"
              % (a, b, 100.0 * (legs[b]["mtp"] - legs[a]["mtp"]) / legs[a]["mtp"]))


if __name__ == "__main__":
    main()
