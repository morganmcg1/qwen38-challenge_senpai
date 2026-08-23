#!/usr/bin/env python3
"""E153 pre-registration arithmetic: frames, mechanism classes, plutarch predictions."""
import json

SRC = "research/board-per-prompt-2026-08-23.json"
REF_ROW = "684821ed"
ROUNDS = {
    "beagle": 110, "botany": 81, "drama": 252, "essays": 92,
    "medicine": 90, "plutarch": 488, "republic": 93, "travel": 213,
}
# Rule 148 median-pair occupancy weights at published score >= 3.5
W = {"beagle": 0.5000, "essays": 0.4474, "republic": 0.0329, "medicine": 0.0197}

R2_DELTA = 154.49869406852432       # E149 Arm C1 ranked-weighted us/round
R2_SE = 14.151340023330699
R1_DELTA_DECODE = 132.55            # E149 Arm A, decode frame, us/round (local)
R1_DELTA_TOTAL = 169.84             # E149 Arm A, total-leg frame, us/round (local)
LOCAL_EDL = 6.3590                  # local benchfixture mean draft steps per round
P_M_GE_6 = 0.5861                   # ranked Rule-148-weighted P(width >= 6)


def main() -> None:
    d = json.load(open(SRC))
    idx = {n: k for k, n in enumerate(d["per_prompt_fields"])}
    row = [x for x in d["rows"] if x["id8"] == REF_ROW][0]
    print(f"reference row {REF_ROW} solver={row['solver']} score={row['officialScore']}")
    print(f"promotedSourceRef={row['promotedSourceRef']}")

    frames = {}
    print()
    hdr = ("prompt", "R", "edl", "nondraft", "dec_us/rd", "tot_us/rd", "dec_pct", "tot_pct")
    print("%-9s %4s %8s %8s %11s %11s %8s %8s" % hdr)
    acc_d = acc_t = 0.0
    for p, v in sorted(row["per_prompt"].items()):
        R = ROUNDS[p]
        mtp = v[idx["mtp_seconds_per_token_mean"]]
        pre = v[idx["prefill_seconds_per_token"]]
        edl = v[idx["effective_mean_draft_len"]]
        nd = v[idx["non_drafting_round_count"]]
        dec = 512 * (mtp - pre) / R * 1e6
        tot = 512 * mtp / R * 1e6
        frames[p] = (R, edl, nd, dec, tot)
        pd_ = R2_DELTA / dec * 100
        pt_ = R2_DELTA / tot * 100
        w = W.get(p, 0.0)
        acc_d += w * pd_
        acc_t += w * pt_
        print("%-9s %4d %8.4f %8d %11.1f %11.1f %8.4f %8.4f"
              % (p, R, edl, nd, dec, tot, pd_, pt_))

    print()
    print("Rule-148 weighted pct for %.2f us/round:" % R2_DELTA)
    print("  decode frame : %.4f pct  -> %.1f us/round per 1 pct" % (acc_d, R2_DELTA / acc_d))
    print("  total leg    : %.4f pct  -> %.1f us/round per 1 pct" % (acc_t, R2_DELTA / acc_t))
    print("  advisor states 0.2946 pct and 524.5 us/round per 1 pct")

    # ---- R1 exposure model -------------------------------------------------
    print()
    print("R1 leaf16: readout fires once per DRAFT STEP (Qwen35.swift:6094/6096 called")
    print("           from Qwen36MTPBlockSession.swift:1633 once + :1649 (draftCount-1) times)")
    ranked_steps = sum(W[p] * frames[p][1] for p in W)
    print("  local steps/round            = %.4f" % LOCAL_EDL)
    print("  Rule-148 ranked steps/round  = %.4f" % ranked_steps)
    print("  depth discount               = %.5f" % (ranked_steps / LOCAL_EDL))
    for name, delta in (("decode", R1_DELTA_DECODE), ("total-leg", R1_DELTA_TOTAL)):
        ranked_us = delta * ranked_steps / LOCAL_EDL
        print("  %-9s frame: %.2f us/round local -> %.2f us/round ranked -> %.4f pct (524.5)"
              % (name, delta, ranked_us, ranked_us / 524.5))
    per_step = R1_DELTA_DECODE / LOCAL_EDL
    print("  us per draft step (decode frame) = %.4f" % per_step)
    plut_R, plut_edl, plut_nd, plut_dec, plut_tot = frames["plutarch"]
    plut_us = per_step * plut_edl
    print("  plutarch: %.4f steps/round x %.4f us = %.4f us/round -> %.5f pct (decode frame)"
          % (plut_edl, per_step, plut_us, plut_us / plut_dec * 100))
    drafting = plut_R - plut_nd
    print("  plutarch exposure ratio round:drafting-round = %.2f" % (plut_R / drafting))
    print("  class contrast on plutarch (decode frame, %.2f us/round local decode):" % R1_DELTA_DECODE)
    for cls, hits in (("per_round", 1.0),
                      ("per_drafting_round", drafting / plut_R),
                      ("per_draft_step", plut_edl / LOCAL_EDL)):
        us = R1_DELTA_DECODE * hits
        print("    %-20s %8.3f us/round  %8.5f pct" % (cls, us, us / plut_dec * 100))

    # ---- R2 exposure model -------------------------------------------------
    print()
    print("R2 merged SDPA: width_gated_at_6 (AttentionUtils.swift:122-124)")
    per_elig = R2_DELTA / P_M_GE_6
    print("  ranked-weighted P(M>=6)      = %.4f" % P_M_GE_6)
    print("  saving per eligible round    = %.2f us" % per_elig)
    lp = plut_edl / 5.0
    print("  plutarch P(M>=6) = P(d>=5) in [0, %.5f]  (LP bound edl/5)" % lp)
    print("  plutarch pct at LP bound     = %.5f pct" % (per_elig * lp / plut_dec * 100))
    print("  plutarch pct point estimate (d>=5 needs 5 consecutive accepts) ~ 0")


if __name__ == "__main__":
    main()
