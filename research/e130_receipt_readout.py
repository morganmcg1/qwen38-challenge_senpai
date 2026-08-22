#!/usr/bin/env python3
"""Read the E130 ranked receipt as an instrument, per advisor note F8.

    usage: research/e130_receipt_readout.py --receipt PREFIX
                                            --reference PREFIX
                                            [--out PATH]

F8 asked for four things back from the E130 receipt, in this order:

1. the full per-prompt `mtp_seconds_per_token_mean` vector, all eight prompts;
2. the F76 mode index, computed and stated, then labelled UNUSABLE for mode
   classification on this candidate;
3. the derived occupancy-to-time coupling `c` on g17s, with its band, from the
   per-prompt deltas against the reference receipt, restricted to the
   head-projection population;
4. `raw_p` for every prompt and not only the published median (Rule 63).

harness=ranked throughout. Every quantity here comes from the trusted parent's
own receipt, so no local-ratio term can leak into it.

WHY THE MODE INDEX IS REPORTED BUT NOT USED
-------------------------------------------
F76's index separates the two hidden-prompt draw modes with a zero-sum
weighting of log candidate decode time. F137 then found that the two receipts
carrying the current frontier draft kernels sit 6 to 8 sd below the historical
fast cluster. A draft-path saving and a slow draw share a per-prompt signature,
so the index cannot tell them apart once the tree contains those kernels. This
candidate contains them. The index is therefore computed and printed for the
record, and is explicitly not used to classify the draw or to correct the
published median.

THE COUPLING MODEL
------------------
The arm changes only which template instantiation the wide-QMV entry point
selects at `case 5`. It deletes no executed instruction on the scored target
path (F121), so its whole claim is residency: 101 registers and 39 simultaneous
groups become 90 and 44, a census-weighted +12.82 % on g17s cells and +0.00 %
on the g16s control.

Only the proposal-head projection population can benefit. F8 retracted F13's
1.82 % and put that share at 7 % to 9 %. So the predicted saving in candidate
seconds per token is

    saving_fraction = s_head * g * c

with g = 0.1282 and c the unknown occupancy-to-time coupling. At s_head = 0.08
a coupling of 1.0 would be worth 1.03 %, which is the scale that makes F8's
"c = 0.05 is worth +0.05 %" and "the g17s bracket 0.199 to 0.301 is worth
+0.20 % to +0.31 %" come out right.

Inverting an observed delta gives c. The band comes from the instrument's own
repeat spread, measured by `research/e130_schedule_family.py` on the four
receipts that carry our exact parent tree. On that tier the relative sd of the
mean candidate decode time is 0.0279 %, while the relative sd of the published
median is 0.1416 %. The published median is 5.1 times the noisier statistic on
one unchanged tree, so it is reported but is not the test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import statistics as st

from e126_modeindex import (
    PER_RUN_NOISE,
    SAME_MODE_SD,
    THRESHOLD,
    WEIGHTS,
    classify,
    index_of,
)

BOARD = pathlib.Path(os.environ.get("YUKON_BOARD_JSON",
                                    "/tmp/yukon-board/full.json"))

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

# E130 census on base 221065c5, replayed and reproduced exactly.
RESIDENCY_GAIN_G17S = 0.1282
RESIDENCY_GAIN_G16S = 0.0000

# F8 retracts F13. Four independent lines put the head-projection share here.
HEAD_SHARE_LO = 0.07
HEAD_SHARE_MID = 0.08
HEAD_SHARE_HI = 0.09

# research/e130_schedule_family.py, tier 0: the four receipts carrying our
# exact parent tree, from three independent solvers.
TIER0_MEMBERS = ("3b376ba2", "c63eaa21", "48423d09", "cf79f7df")
TIER0_CANDIDATE_REL_SD_PCT = 0.0279
TIER0_SCORE_REL_SD_PCT = 0.1416
TIER0_N = 4

# research/e130_instrument_sd.py, identification B: the same quantity pooled
# over 36 repeat tiers, 70 degrees of freedom, 95 % interval
# [0.0487, 0.0699]. Pooling includes tiers that may mix near-identical trees,
# so this is the conservative estimate and tier 0 alone is the optimistic one.
POOLED_CANDIDATE_REL_SD_PCT = 0.0532
POOLED_CANDIDATE_REL_SD_CI = (0.0487, 0.0699)
POOLED_SCORE_REL_SD_PCT = 0.2546
POOLED_DOF = 70

# research/e130_instrument_sd.py, identification C. The serial leg is a pinned
# prebuilt binary in the runner-owned workspace, so no candidate edit moves it.
SERIAL_LEG_REL_SD_PCT = 0.1113

# The bracket F8 asked this receipt to test.
BRACKET_LO, BRACKET_HI = 0.199, 0.301


def load_receipt(prefix: str) -> dict:
    rows = json.loads(BOARD.read_text())
    hits = [r for r in rows if str(r.get("id", "")).startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit("%r matched %d board rows" % (prefix, len(hits)))
    return hits[0]


def per_prompt(row: dict) -> dict[str, dict]:
    out = {}
    for entry in (row.get("officialMetrics") or {}).get("per_prompt") or []:
        name = PROMPT_NAMES.get(str(entry.get("prompt_sha256", ""))[:8])
        if name:
            out[name] = entry
    if len(out) != 8:
        raise SystemExit("receipt %s has %d named prompts, not 8"
                         % (row["id"][:8], len(out)))
    return out


def published_median(raw: list[float]) -> float:
    ordered = sorted(raw)
    return 0.5 * (ordered[3] + ordered[4])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    cand_row = load_receipt(args.receipt)
    ref_row = load_receipt(args.reference)
    if cand_row.get("officialScore") is None:
        raise SystemExit("receipt %s has no officialScore yet (status=%s)"
                         % (cand_row["id"][:8], cand_row.get("status")))

    cand, ref = per_prompt(cand_row), per_prompt(ref_row)
    names = sorted(WEIGHTS)
    report: dict = {
        "harness": "ranked",
        "receipt": cand_row["id"],
        "receipt_status": cand_row.get("status"),
        "receipt_score": float(cand_row["officialScore"]),
        "reference": ref_row["id"],
        "reference_solver": ref_row.get("solverUsername"),
        "reference_score": float(ref_row["officialScore"]),
    }

    # ---- deliverable 1 and 4: the full per-prompt vector, with raw_p ----
    print("=" * 78)
    print("1 and 4. PER-PROMPT VECTOR AND raw_p            harness=ranked")
    print("=" * 78)
    print("receipt   %s  score %.8f  status %s"
          % (cand_row["id"][:8], report["receipt_score"],
             cand_row.get("status")))
    print("reference %s  score %.8f  solver %s"
          % (ref_row["id"][:8], report["reference_score"],
             ref_row.get("solverUsername")))
    print()
    head = ("%-9s %14s %14s %11s %11s %9s %6s"
            % ("prompt", "mtp_spt", "serial_spt", "raw_p", "ref raw_p",
               "draftlen", "par"))
    print(head)
    print("-" * len(head))
    rows = []
    for name in names:
        c, r = cand[name], ref[name]
        rows.append({
            "prompt": name,
            "mtp_seconds_per_token_mean": c["mtp_seconds_per_token_mean"],
            "serial_seconds_per_token_mean": c["serial_seconds_per_token_mean"],
            "raw_ratio_of_means": c["raw_ratio_of_means"],
            "reference_raw_ratio_of_means": r["raw_ratio_of_means"],
            "reference_mtp_seconds_per_token_mean":
                r["mtp_seconds_per_token_mean"],
            "effective_mean_draft_len": c["effective_mean_draft_len"],
            "parity_ok": c["parity_ok"],
            "head_provenance_sha256": c["head_provenance_sha256"],
            "non_drafting_round_count": c["non_drafting_round_count"],
        })
        print("%-9s %14.10f %14.10f %11.6f %11.6f %9.4f %6s"
              % (name, c["mtp_seconds_per_token_mean"],
                 c["serial_seconds_per_token_mean"],
                 c["raw_ratio_of_means"], r["raw_ratio_of_means"],
                 c["effective_mean_draft_len"], c["parity_ok"]))
    report["per_prompt"] = rows

    raw = [r["raw_ratio_of_means"] for r in rows]
    ref_raw = [r["reference_raw_ratio_of_means"] for r in rows]
    med, ref_med = published_median(raw), published_median(ref_raw)
    print()
    print("raw_p sorted   %s" % " ".join("%.5f" % x for x in sorted(raw)))
    print("median rule    mean of the 4th and 5th of eight sorted")
    print("published      %.8f   (receipt states %.8f)"
          % (med, report["receipt_score"]))
    print("reference      %.8f" % ref_med)
    print("spread of raw_p across prompts: min %.5f  max %.5f  rel sd %.4f %%"
          % (min(raw), max(raw), 100.0 * st.stdev(raw) / st.mean(raw)))
    print("Rule 63: the median keeps two of eight prompts and mixes in a fresh")
    print("serial draw, so it is the weakest reading available here.")
    report["published_median_recomputed"] = med
    report["reference_median_recomputed"] = ref_med

    # ---- deliverable 2: the mode index, computed then disqualified ----
    print()
    print("=" * 78)
    print("2. F76 MODE INDEX: COMPUTED, THEN UNUSABLE")
    print("=" * 78)
    cand_times = {n: cand[n]["mtp_seconds_per_token_mean"] for n in names}
    ref_times = {n: ref[n]["mtp_seconds_per_token_mean"] for n in names}
    idx, ref_idx = index_of(cand_times), index_of(ref_times)
    label, dist = classify(idx)
    ref_label, ref_dist = classify(ref_idx)
    print("index(candidate) = %+.4f   nominal label %-4s  %.2f same-mode sd "
          "from the %.1f threshold" % (idx, label, dist, THRESHOLD))
    print("index(reference) = %+.4f   nominal label %-4s  %.2f same-mode sd"
          % (ref_idx, ref_label, ref_dist))
    print("difference       = %+.4f   (one mode flip is 1.000 index units, "
          "same-mode sd %.3f, per-run noise %.4f)"
          % (idx - ref_idx, SAME_MODE_SD, PER_RUN_NOISE))
    print()
    print("UNUSABLE FOR MODE CLASSIFICATION ON THIS CANDIDATE.")
    print("F137 found that both receipts carrying the current frontier draft")
    print("kernels sit 6 to 8 sd below the historical fast cluster. A saving on")
    print("the draft path and a slow draw move the same per-prompt weights in")
    print("the same direction, so the index cannot separate them once the tree")
    print("contains those kernels. This candidate contains them: it is the")
    print("reference tree plus a two-line entry-point change. The index above")
    print("is recorded for the ledger only. It is not used to classify this")
    print("draw and it is not used to correct the published median.")
    report["mode_index"] = {
        "candidate": idx,
        "reference": ref_idx,
        "difference": idx - ref_idx,
        "nominal_label_candidate": label,
        "nominal_label_reference": ref_label,
        "threshold": THRESHOLD,
        "usable_for_mode_classification": False,
        "why_unusable": "F137: with the frontier draft kernels present, a "
                        "draft-path saving and a slow draw share a per-prompt "
                        "signature, so the index is confounded.",
    }

    # ---- deliverable 3: the derived coupling c, with its band ----
    print()
    print("=" * 78)
    print("3. DERIVED OCCUPANCY-TO-TIME COUPLING c ON g17s")
    print("=" * 78)
    print("paired per-prompt change in candidate decode time.")
    print("negative means this receipt was FASTER than the reference.\n")
    print("%-9s %14s %14s %10s" % ("prompt", "cand mtp_spt", "ref mtp_spt",
                                   "delta %"))
    print("-" * 50)
    deltas = []
    for name in names:
        d = 100.0 * (cand_times[name] - ref_times[name]) / ref_times[name]
        deltas.append(d)
        print("%-9s %14.10f %14.10f %+10.4f"
              % (name, cand_times[name], ref_times[name], d))
    mean_delta = st.mean(deltas)
    print("-" * 50)
    print("mean delta over eight prompts   %+.4f %%" % mean_delta)
    print("sd of the eight deltas          %.4f %%" % st.stdev(deltas))
    print("empirical se of the mean        %.4f %%"
          % (st.stdev(deltas) / math.sqrt(8.0)))
    print("\nThat pairing uses one reference run. The stronger test compares")
    print("this receipt with the whole tier of receipts carrying our exact")
    print("parent tree, so the reference is an average and not a single draw.")

    # The test: this receipt against the tier that shares our parent tree.
    tier_rows = [load_receipt(p) for p in TIER0_MEMBERS]
    tier_means = [st.mean([per_prompt(r)[n]["mtp_seconds_per_token_mean"]
                           for n in names]) for r in tier_rows]
    tier_mean = st.mean(tier_means)
    cand_mean = st.mean([cand_times[n] for n in names])
    observed_delta_pct = 100.0 * (cand_mean - tier_mean) / tier_mean
    observed_saving_pct = -observed_delta_pct

    # One new draw against a mean of TIER0_N draws. The conservative sd is
    # pooled over 36 tiers and 70 dof; the optimistic one is tier 0 alone.
    spread = math.sqrt(1.0 + 1.0 / TIER0_N)
    se_conservative = POOLED_CANDIDATE_REL_SD_PCT * spread
    se_pt = TIER0_CANDIDATE_REL_SD_PCT * spread
    se_ci = tuple(x * spread for x in POOLED_CANDIDATE_REL_SD_CI)

    print()
    print("TIER TEST. mean candidate decode time, seconds per token.")
    print("  parent tier   %s" % " ".join(TIER0_MEMBERS))
    for prefix, value in zip(TIER0_MEMBERS, tier_means):
        print("    %-9s %.8f" % (prefix, value))
    print("  tier mean     %.8f   tier rel sd %.4f %% on 3 dof"
          % (tier_mean, TIER0_CANDIDATE_REL_SD_PCT))
    print("  this receipt  %.8f" % cand_mean)
    print("  delta         %+.4f %%   saving %+.4f %%"
          % (observed_delta_pct, observed_saving_pct))
    print("  se of one new draw against the %d-run tier mean:" % TIER0_N)
    print("    %.4f %%  conservative, pooled sd %.4f %% on %d dof, "
          "95 %% [%.4f, %.4f]"
          % (se_conservative, POOLED_CANDIDATE_REL_SD_PCT, POOLED_DOF,
             se_ci[0], se_ci[1]))
    print("    %.4f %%  optimistic, tier 0 sd alone" % se_pt)
    print("  z = %+.2f conservative, %+.2f optimistic"
          % (observed_saving_pct / se_conservative,
             observed_saving_pct / se_pt))

    print("\nc = saving / (s_head * g),  g = %.4f on g17s, %.4f on g16s"
          % (RESIDENCY_GAIN_G17S, RESIDENCY_GAIN_G16S))
    couplings = {}
    for tag, share in (("lo", HEAD_SHARE_LO), ("mid", HEAD_SHARE_MID),
                       ("hi", HEAD_SHARE_HI)):
        scale = 100.0 * share * RESIDENCY_GAIN_G17S  # percent worth of c = 1
        point = observed_saving_pct / scale
        band = tuple(sorted((
            (observed_saving_pct - 2.0 * se_conservative) / scale,
            (observed_saving_pct + 2.0 * se_conservative) / scale)))
        tight = tuple(sorted((
            (observed_saving_pct - 2.0 * se_pt) / scale,
            (observed_saving_pct + 2.0 * se_pt) / scale)))
        couplings[tag] = {
            "head_share": share,
            "percent_worth_of_c_equals_1": scale,
            "c_point": point,
            "c_band_2se_conservative": list(band),
            "c_band_2se_point": list(tight),
        }
        print("  s_head=%.2f  c = 1 is worth %.4f %%   c = %+.4f   "
              "2se band [%+.3f, %+.3f] point, [%+.3f, %+.3f] conservative"
              % (share, scale, point, tight[0], tight[1], band[0], band[1]))

    mid = couplings["mid"]
    bracket_rejected_pt = mid["c_band_2se_point"][1] < BRACKET_LO
    bracket_rejected_cons = mid["c_band_2se_conservative"][1] < BRACKET_LO
    print()
    print("F8 bracket for g17s: c in [%.3f, %.3f], worth %+.3f %% to %+.3f %%"
          % (BRACKET_LO, BRACKET_HI,
             100.0 * HEAD_SHARE_MID * RESIDENCY_GAIN_G17S * BRACKET_LO,
             100.0 * HEAD_SHARE_MID * RESIDENCY_GAIN_G17S * BRACKET_HI))
    print("  bracket bottom rejected at 2 se, point sd:        %s"
          % bracket_rejected_pt)
    print("  bracket bottom rejected at 2 se, conservative sd: %s"
          % bracket_rejected_cons)
    scale_mid = mid["percent_worth_of_c_equals_1"]
    sep_pt = 2.0 * se_pt / scale_mid
    sep_cons = 2.0 * se_conservative / scale_mid
    gap = 100.0 * HEAD_SHARE_MID * RESIDENCY_GAIN_G17S * 0.14
    print()
    print("RESOLUTION OF ONE RECEIPT. smallest separable difference in c at "
          "2 se:")
    print("  %.3f in c conservative, %.3f optimistic." % (sep_cons, sep_pt))
    print("  c = 0.01 against c = 0.15 is a gap of 0.140 in c, or %.3f %% of"
          % gap)
    print("  candidate decode time.")
    se_median = POOLED_SCORE_REL_SD_PCT * spread
    print("  F8 expected that one receipt could not separate those two values.")
    print("  Read on the PUBLISHED MEDIAN that is correct: its pooled rel sd")
    print("  is %.4f %%, so the gap is only %.1f se there."
          % (POOLED_SCORE_REL_SD_PCT, gap / se_median))
    print("  Read on MEAN CANDIDATE DECODE TIME, which the receipt also")
    print("  publishes, the same gap is %.1f se conservative and %.1f se"
          % (gap / se_conservative, gap / se_pt))
    print("  optimistic. The statistic, not the run count, was the limit.")
    report["coupling"] = {
        "residency_gain_g17s": RESIDENCY_GAIN_G17S,
        "residency_gain_g16s": RESIDENCY_GAIN_G16S,
        "per_prompt_delta_pct_vs_reference": dict(zip(names, deltas)),
        "mean_delta_pct_vs_reference": mean_delta,
        "tier_members": list(TIER0_MEMBERS),
        "tier_candidate_means": tier_means,
        "tier_mean_candidate_spt": tier_mean,
        "receipt_mean_candidate_spt": cand_mean,
        "observed_delta_pct_vs_tier": observed_delta_pct,
        "observed_saving_pct": observed_saving_pct,
        "se_optimistic_pct": se_pt,
        "se_conservative_pct": se_conservative,
        "se_conservative_ci95_pct": list(se_ci),
        "z_optimistic": observed_saving_pct / se_pt,
        "z_conservative": observed_saving_pct / se_conservative,
        "by_head_share": couplings,
        "f8_bracket": [BRACKET_LO, BRACKET_HI],
        "f8_bracket_bottom_rejected_optimistic": bracket_rejected_pt,
        "f8_bracket_bottom_rejected_conservative": bracket_rejected_cons,
        "smallest_separable_c_at_2se_optimistic": sep_pt,
        "smallest_separable_c_at_2se_conservative": sep_cons,
        "c_0p01_vs_c_0p15_sigma_on_mean_candidate_time_conservative":
            gap / se_conservative,
        "c_0p01_vs_c_0p15_sigma_on_mean_candidate_time_optimistic":
            gap / se_pt,
        "c_0p01_vs_c_0p15_sigma_on_published_median": gap / se_median,
    }

    # ---- the serial null ----
    print()
    print("=" * 78)
    print("SERIAL NULL: what a candidate edit provably cannot move")
    print("=" * 78)
    ser_deltas = [100.0 * (cand[n]["serial_seconds_per_token_mean"]
                           - ref[n]["serial_seconds_per_token_mean"])
                  / ref[n]["serial_seconds_per_token_mean"] for n in names]
    print("mean %+.4f %%   sd %.4f %%   min %+.4f %%   max %+.4f %%"
          % (st.mean(ser_deltas), st.stdev(ser_deltas), min(ser_deltas),
             max(ser_deltas)))
    print("The serial leg runs a pinned prebuilt binary in the runner-owned")
    print("baseline workspace, so this row is pure instrument noise. Any")
    print("candidate delta smaller than this is not a measurement of the arm.")
    report["serial_null_pct"] = {
        "per_prompt": dict(zip(names, ser_deltas)),
        "mean": st.mean(ser_deltas),
        "sd": st.stdev(ser_deltas),
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
