#!/usr/bin/env python3
"""Attribute the measured 4->3 draft-readout gain to its causes.

`draft_bits_phase3.py` reports the model-free split work = rows x cost-per-row.
That split is exact but it does not say how much of the gain is the bandwidth
mechanism, because `cost-per-row` also absorbs the per-round fixed cost and the
slightly different draft-calls-per-row ratio.

This script anchors the mechanism term on the independently measured Phase 1
readout cost, which lets the leftover be named honestly as a draft-trajectory
windfall on one public prompt. It also models what changes if the drafting mix
matches the ranked 4-bit-head configuration instead of the locally observed one.
"""
import argparse
import json
import os
import statistics as st

# Phase 1 (this host, tag e15r2-p1-b85e782): single-row compact-readout matmul.
P1_SECONDS_PER_CALL = {4: 0.0011650, 3: 0.0008819, 2: 0.0006709}

# research/ESTABLISHED_FACTS.md: per-draft-step bytes. The compact readout is
# shared and does NOT shrink with head precision, so only the head term moves.
READOUT_MB_AT_4BIT = 283.2
HEAD_MB = {"bf16": 849.4, "q4": 238.9}


def arm(root, prefix, slot, bits):
    d = os.path.join(root, "%s-p%d-b%s" % (prefix, slot, bits))
    return (json.load(open(os.path.join(d, "reports/04-mtp-timed.json"))),
            json.load(open(os.path.join(d, "amdahl.json"))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--order", default="4,3,3,4")
    ap.add_argument("--root", default=".mlxfast-private/draft-bits")
    ap.add_argument("--control", type=int, default=4)
    ap.add_argument("--candidate", type=int, default=3)
    args = ap.parse_args()
    order = [int(b) for b in args.order.split(",")]

    arms = {}
    for slot, bits in enumerate(order, 1):
        arms.setdefault(bits, []).append(arm(args.root, args.prefix, slot, bits))

    agg = {}
    for bits, reps in arms.items():
        t = [r for r, _ in reps]
        a = [x for _, x in reps]
        rows = {r["declared_rows_total"] for r in t}
        rounds = {r["round_count"] for r in t}
        assert len(rows) == 1 and len(rounds) == 1, "work varied across replicates"
        agg[bits] = {
            "replicates": len(reps),
            "rows": rows.pop(),
            "rounds": rounds.pop(),
            "accept_rate": t[0]["accepted_draft_rate"],
            "work": st.mean(x["amdahl"]["decode_work_mtp_seconds"] for x in a),
            "prefill": st.mean(x["amdahl"]["prefill_seconds"] for x in a),
            "leg": st.mean(x["amdahl"]["ranked_window_leg_seconds_mtp"] for x in a),
            "score": st.mean(x["amdahl"]["measured_local_score"] for x in a),
            "mtp_spt": st.mean(
                x["mtp_leg"]["parent_measured_seconds_per_token"] for x in a),
            "serial_spt": st.mean(
                x["serial_leg"]["parent_measured_seconds_per_token"] for x in a),
        }
        agg[bits]["calls"] = agg[bits]["rows"] - agg[bits]["rounds"]

    c, k = agg[args.control], agg[args.candidate]
    print("== arms (mean of replicates) ==")
    for bits in (args.control, args.candidate):
        a = agg[bits]
        print("  bits=%d n=%d rounds=%d rows=%d draft_calls=%d accept=%.4f "
              "work=%.6fs prefill=%.6fs leg=%.6fs score=%.7f"
              % (bits, a["replicates"], a["rounds"], a["rows"], a["calls"],
                 a["accept_rate"], a["work"], a["prefill"], a["leg"], a["score"]))

    d_work = k["work"] - c["work"]
    print("\n== Phase-1-anchored attribution of the decode-work delta ==")
    print("  measured delta                  %+.3f ms  (%+.4f%%)"
          % (1e3 * d_work, 100.0 * d_work / c["work"]))
    prec = k["calls"] * (P1_SECONDS_PER_CALL[args.candidate]
                         - P1_SECONDS_PER_CALL[args.control])
    fewer = -(c["calls"] - k["calls"]) * P1_SECONDS_PER_CALL[args.control]
    rest = d_work - prec - fewer
    for name, v in (("readout precision (mechanism)", prec),
                    ("fewer draft calls", fewer),
                    ("fewer rows/rounds (trajectory)", rest)):
        print("    %-32s %+8.3f ms  %5.1f%% of the delta"
              % (name, 1e3 * v, 100.0 * v / d_work))
    print("  mechanism as %% of decode work    %+.4f%%   "
          "(Phase 1 predicted -1.00%% a priori)" % (100.0 * prec / c["work"]))

    conv = 1.0 - c["prefill"] / c["leg"]
    print("\n== score currency ==")
    print("  prefill is %.2f%% of the control MTP leg => conversion factor %.5f"
          % (100.0 * c["prefill"] / c["leg"], conv))
    obs_leg = (k["leg"] - c["leg"]) / c["leg"]
    obs_score = k["score"] / c["score"] - 1.0
    mech_score = 1.0 / (1.0 + conv * prec / c["work"]) - 1.0
    print("  observed  leg %+.4f%%  ->  local score %.7f -> %.7f  (%+.4f%%)"
          % (100.0 * obs_leg, c["score"], k["score"], 100.0 * obs_score))
    print("  mechanism only, trajectory removed:      score %+.4f%%"
          % (100.0 * mech_score))

    print("\n== drafting-mix transfer model (NOT a measurement) ==")
    bw = READOUT_MB_AT_4BIT / P1_SECONDS_PER_CALL[args.control] / 1e3
    print("  measured readout bandwidth on this host  %.1f GB/s" % bw)
    verify = c["work"] - c["calls"] * (HEAD_MB["bf16"] + READOUT_MB_AT_4BIT) / bw / 1e3
    for label, mb in (("observed local (bf16-geometry head)", HEAD_MB["bf16"]),
                      ("ranked declared 4-bit head", HEAD_MB["q4"])):
        step = (mb + READOUT_MB_AT_4BIT) / bw / 1e3
        work = verify + c["calls"] * step
        share = READOUT_MB_AT_4BIT / (mb + READOUT_MB_AT_4BIT)
        leg = c["prefill"] + work
        cv = 1.0 - c["prefill"] / leg
        print("    %-36s readout=%.2f%% of draft bytes  step=%.3f ms  "
              "work=%.3fs  mech=%+.4f%% work  %+.4f%% score"
              % (label, 100.0 * share, 1e3 * step, work,
                 100.0 * prec / work,
                 100.0 * (1.0 / (1.0 + cv * prec / work) - 1.0)))
    print("  verify+overhead residual (head-independent) %.3fs" % verify)


if __name__ == "__main__":
    main()
