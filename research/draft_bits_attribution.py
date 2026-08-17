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


def aggregate(root, prefix, order):
    arms = {}
    for slot, bits in enumerate(order, 1):
        arms.setdefault(bits, []).append(arm(root, prefix, slot, bits))

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
    return agg


def attribute(agg, control=4, candidate=3):
    """Split the decode-work delta into mechanism, call-count, and trajectory.

    The mechanism term is priced from Phase 1's independently measured readout
    cost rather than fitted here, so the leftover cannot silently absorb it.
    """
    c, k = agg[control], agg[candidate]
    d_work = k["work"] - c["work"]
    prec = k["calls"] * (P1_SECONDS_PER_CALL[candidate]
                         - P1_SECONDS_PER_CALL[control])
    fewer = -(c["calls"] - k["calls"]) * P1_SECONDS_PER_CALL[control]
    rest = d_work - prec - fewer
    conv = 1.0 - c["prefill"] / c["leg"]

    out = {
        "control_bits": control,
        "candidate_bits": candidate,
        "decode_work_delta_seconds": d_work,
        "decode_work_delta_pct": 100.0 * d_work / c["work"],
        "term_readout_precision_seconds": prec,
        "term_fewer_draft_calls_seconds": fewer,
        "term_trajectory_seconds": rest,
        "term_readout_precision_share_pct": 100.0 * prec / d_work,
        "term_fewer_draft_calls_share_pct": 100.0 * fewer / d_work,
        "term_trajectory_share_pct": 100.0 * rest / d_work,
        "mechanism_pct_of_decode_work": 100.0 * prec / c["work"],
        "phase1_predicted_pct_of_decode_work": -1.00,
        "prefill_share_of_control_leg_pct": 100.0 * c["prefill"] / c["leg"],
        "score_conversion_factor": conv,
        "observed_leg_pct": 100.0 * (k["leg"] - c["leg"]) / c["leg"],
        "observed_score_pct": 100.0 * (k["score"] / c["score"] - 1.0),
        "mechanism_only_score_pct":
            100.0 * (1.0 / (1.0 + conv * prec / c["work"]) - 1.0),
    }

    bw = READOUT_MB_AT_4BIT / P1_SECONDS_PER_CALL[control] / 1e3
    verify = c["work"] - c["calls"] * (
        HEAD_MB["bf16"] + READOUT_MB_AT_4BIT) / bw / 1e3
    out["readout_bandwidth_gbps"] = bw
    out["verify_overhead_residual_seconds"] = verify
    out["transfer"] = []
    for label, key in (("observed local (bf16-geometry head)", "bf16"),
                       ("ranked declared 4-bit head", "q4")):
        mb = HEAD_MB[key]
        step = (mb + READOUT_MB_AT_4BIT) / bw / 1e3
        work = verify + c["calls"] * step
        leg = c["prefill"] + work
        cv = 1.0 - c["prefill"] / leg
        out["transfer"].append({
            "label": label,
            "head": key,
            "head_mb": mb,
            "readout_share_of_draft_bytes_pct":
                100.0 * READOUT_MB_AT_4BIT / (mb + READOUT_MB_AT_4BIT),
            "draft_step_ms": 1e3 * step,
            "decode_work_seconds": work,
            "mechanism_pct_of_decode_work": 100.0 * prec / work,
            "mechanism_score_pct":
                100.0 * (1.0 / (1.0 + cv * prec / work) - 1.0),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--order", default="4,3,3,4")
    ap.add_argument("--root", default=".mlxfast-private/draft-bits")
    ap.add_argument("--control", type=int, default=4)
    ap.add_argument("--candidate", type=int, default=3)
    args = ap.parse_args()
    order = [int(b) for b in args.order.split(",")]

    agg = aggregate(args.root, args.prefix, order)
    a = attribute(agg, args.control, args.candidate)
    c = agg[args.control]

    print("== arms (mean of replicates) ==")
    for bits in (args.control, args.candidate):
        x = agg[bits]
        print("  bits=%d n=%d rounds=%d rows=%d draft_calls=%d accept=%.4f "
              "work=%.6fs prefill=%.6fs leg=%.6fs score=%.7f"
              % (bits, x["replicates"], x["rounds"], x["rows"], x["calls"],
                 x["accept_rate"], x["work"], x["prefill"], x["leg"], x["score"]))

    print("\n== Phase-1-anchored attribution of the decode-work delta ==")
    print("  measured delta                  %+.3f ms  (%+.4f%%)"
          % (1e3 * a["decode_work_delta_seconds"], a["decode_work_delta_pct"]))
    for name, key in (("readout precision (mechanism)", "readout_precision"),
                      ("fewer draft calls", "fewer_draft_calls"),
                      ("fewer rows/rounds (trajectory)", "trajectory")):
        print("    %-32s %+8.3f ms  %5.1f%% of the delta"
              % (name, 1e3 * a["term_%s_seconds" % key],
                 a["term_%s_share_pct" % key]))
    print("  mechanism as %% of decode work    %+.4f%%   "
          "(Phase 1 predicted %.2f%% a priori)"
          % (a["mechanism_pct_of_decode_work"],
             a["phase1_predicted_pct_of_decode_work"]))

    print("\n== score currency ==")
    print("  prefill is %.2f%% of the control MTP leg => conversion factor %.5f"
          % (a["prefill_share_of_control_leg_pct"], a["score_conversion_factor"]))
    print("  observed  leg %+.4f%%  ->  local score %.7f -> %.7f  (%+.4f%%)"
          % (a["observed_leg_pct"], c["score"], agg[args.candidate]["score"],
             a["observed_score_pct"]))
    print("  mechanism only, trajectory removed:      score %+.4f%%"
          % a["mechanism_only_score_pct"])

    print("\n== drafting-mix transfer model (NOT a measurement) ==")
    print("  measured readout bandwidth on this host  %.1f GB/s"
          % a["readout_bandwidth_gbps"])
    for t in a["transfer"]:
        print("    %-36s readout=%.2f%% of draft bytes  step=%.3f ms  "
              "work=%.3fs  mech=%+.4f%% work  %+.4f%% score"
              % (t["label"], t["readout_share_of_draft_bytes_pct"],
                 t["draft_step_ms"], t["decode_work_seconds"],
                 t["mechanism_pct_of_decode_work"], t["mechanism_score_pct"]))
    print("  verify+overhead residual (head-independent) %.3fs"
          % a["verify_overhead_residual_seconds"])


if __name__ == "__main__":
    main()
