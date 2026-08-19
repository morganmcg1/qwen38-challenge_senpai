#!/usr/bin/env python3
"""E49 pre-registration. Committed before the first GPU second of this experiment.

Ledger 173(C) values a 2-stream M=9 at +5.36 % of score (7.0 sd) by combining
E46's refit with askeladd's width histogram. The cell it values, `<T,9,5>`, has
never been timed. E49 times it, and splits the E27 residual into the two parts
that have opposite consequences:

  H_local_eaten   `<T,9,5>` is not faster than `<T,9,3>` even in isolation. The
                  +21 registers are paid inside the same cell, there is no
                  prize, and no route to it is worth building.
  H_shared_tax    `<T,9,5>` IS ~15 % faster in isolation, and E27 lost score
                  because putting it in the one shared `[[kernel]]` raised the
                  allocation for every width and taxed the untouched ones.

Arm 1 (isolated) measures the first. Arm 2 (a dose ladder on the kernel-wide
allocation, with no dispatched width's code changed) measures the second.

Everything here -- predictions, decision thresholds, the noise bar, and what
each outcome means -- is fixed before the first timed leg. `research/e49_analyze.py`
applies these constants without re-deriving any of them.

  python3 research/e49_prereg.py --out research/e49-prereg.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parent.parent

BASE_SHA = "fb0a09d3912477d94ed631bdb90fd04172d7b4cf"
ASSIGNMENT = "qwen38-r1-e49-m9-two-stream-local-vs-shared"

# --- E46's refit on the shipped NA<=4 table, quoted with its tree -------------
# research/qwen38-r1-e46-stream-vs-groupwidth.md, PR #51, base 01f69e18.
# T(M) = C + B*streams(M) + A*M, max|resid| 0.770 ms over 73.5-186.1 ms.
E46_A = 9.624    # ms per input row
E46_B = 27.532   # ms per marginal weight stream
E46_C = 16.757
E46_T9_MEASURED = 186.098  # ms, mean of the two E46 base legs, same harness
# E46 contrast B: M=8, 3 streams vs 2 streams, +18.72 % for the extra stream.
E46_CONTRAST_B_RATIO = 1.1872

# --- Arm 1 predictions --------------------------------------------------------
_pred_t = lambda m, s: E46_C + E46_B * s + E46_A * m  # noqa: E731
ARM1 = {
    "contrast": "<T,9,5> (2 streams) vs <T,9,3> (3 streams), isolated kernels",
    "isolation": "only case 9 survives in the >=4096 crossrow switch, so the "
                 "cell under test is the only crossrow body the shared "
                 "allocation must cover. Census: entry heuristic 135 (iso3) vs "
                 "181 (iso5), against 163 for the shipped table.",
    "predicted_pct_from_e46_refit": round(
        100.0 * (_pred_t(9, 2) / _pred_t(9, 3) - 1.0), 2),
    "predicted_pct_from_contrast_b": round(
        100.0 * (1.0 / E46_CONTRAST_B_RATIO - 1.0), 2),
    "predicted_ms_from_e46_refit": round(_pred_t(9, 2) - _pred_t(9, 3), 3),
    "measured_reference_t9_ipg3_ms": E46_T9_MEASURED,
    "readout": "T(9) = sum over the 8 scored shapes of "
               "calls_per_verify * seconds_per_call, ms, identical to E46.",
}

# --- Arm 2 predictions --------------------------------------------------------
# The advisor solves (1+c_local)(1+c_ceiling) = 1+c_net with c_local = -9.134 %
# and c_net = +0.4947 %, giving c_ceiling = +10.6 % of QMV cost on widths whose
# code did not change. Occupancy is a step function, so that is an INTEGRAL
# across 108 -> 129, not a slope; the ladder exists to see ramp vs step.
ARM2 = {
    "prediction_pct_at_129": 10.6,
    "prediction_kind": "integral across 108 -> 129, not a per-register slope",
    "dose_knob": "an unreachable `case 10:` in the >=4096 switch. ntg.x == M "
                 "and the workload never verifies more than 9 rows, so the cell "
                 "is compiled and allocated for but never executed: no "
                 "dispatched width's instructions change.",
    "doses": [
        {"arm": "shipped", "cell_max": 108, "entry_heuristic": 163,
         "role": "control"},
        {"arm": "dose_null", "cell_max": 108, "entry_heuristic": 164,
         "cell": "<T,4,4> = 104 regs",
         "role": "null: adds a branch WITHOUT raising the max. Reproduces E38 "
                 "arm (b)'s null (117 under a 129 ceiling, no tax). Predicted "
                 "0 %; it is also the contention detector, because a peer "
                 "student's GPU work would inflate this arm exactly like a "
                 "real tax."},
        {"arm": "dose_129", "cell_max": 129, "entry_heuristic": 181,
         "cell": "<T,9,5> = 129 regs",
         "role": "E27's exact ceiling. Predicted +10.6 %."},
        {"arm": "dose_big", "cell_max": 144, "entry_heuristic": 197,
         "cell": "<T,12,6> = 144 regs",
         "role": "dose-response: if the 108->129 interval contains a step, a "
                 "larger dose must not be smaller."},
        {"arm": "dose_huge", "cell_max": 177, "entry_heuristic": 230,
         "cell": "<T,16,8> = 177 regs",
         "role": "a dose we cannot have missed. A null here refutes the "
                 "mechanism far more strongly than a null at 129 alone."},
        {"arm": "e27_replica", "cell_max": 129, "entry_heuristic": 181,
         "role": "composite: E27's own M=9 edit. Falsifies "
                 "composite == local + tax rather than assuming it."},
    ],
    "untouched_widths": [3, 4, 5, 6, 7, 8, 9],
    "ordering": "palindromic within a session so every arm has the same mean "
                "sweep position and linear thermal drift cancels; the null is "
                "interleaved between positive doses, never run once at the start.",
}

# --- decision rules -----------------------------------------------------------
RULES = {
    "effect": "delta_pct(M) = 100 * (T_treated(M) / T_control(M) - 1)",
    "replicate_floor": "MDE(M) = max(|T_a1 - T_a2|, |T_b1 - T_b2|) over the "
                       "same-build replicates, as E46. With n=2 the honest "
                       "floor is the observed same-build spread, not a "
                       "t-statistic.",
    "control_bar": "the largest |delta_pct| on any width whose code is "
                   "byte-identical between the two builds. E46's worst "
                   "untreated control was +6.41 % at M=1 (a warmup width) and "
                   "<=0.4 % everywhere else.",
    "arm1_local_win": "delta_pct(9) <= -8 % => the 2-stream prize is real in "
                      "isolation; run arm 2.",
    "arm1_eaten": "delta_pct(9) >= -2 % => H_local_eaten. STOP. Do not run arm "
                  "2; report that ledger 173(C)'s largest roadmap item does "
                  "not exist.",
    "arm1_middle": "-8 % < delta_pct(9) < -2 % => a partial local win. Report "
                   "the interval and still run arm 2, because the tax question "
                   "stays decision-relevant at any non-zero local win.",
    "arm2_confirmed": "pooled tax over untouched widths >= +8 % at dose_129 => "
                      "173(C)'s attribution is confirmed. STOP; the follow-up "
                      "is a design problem, not a measurement.",
    "arm2_refuted": "|pooled tax| <= 2 % at dose_129 AND at the larger doses "
                    "=> 173(C) is refuted and E27's loss is something else.",
    "arm2_invalid": "|tax| > 2 % on dose_null invalidates the ladder: either "
                    "the harness moves under a recompile alone, or something "
                    "else was on the GPU. Report and do not interpret the "
                    "positive doses.",
    "no_significance_chasing": "reps buy precision, only the design buys power. "
                               "No arm is extended to chase a p-value.",
}

INSTRUMENT = {
    "register_readout": "There is no true register or occupancy readout on this "
                        "box (ledger 157). peak_live_regs is a textual "
                        "peak-live-SSA heuristic; it is used ONLY to choose and "
                        "order the doses, never as a result.",
    "timing_readout": "research/run-qmv-curve.sh, the same harness and the same "
                      "T(M) definition as E46, so E46's numbers are directly "
                      "comparable.",
    "thermal": "This host's idle GPU floor is ~43.2 C, above benchmark.sh's 40 C "
               "target, so every leg records cool_gate_passed_real_gate=false "
               "and gate_qualified_for_timing=false. Directional causal "
               "evidence inside one counterbalanced session only; never a "
               "ranked or gate-qualified score.",
    "contention": "MLXFAST_LOCAL_RUN_LOCK_DIR=/tmp/mlxfast-shared so the run "
                  "lock is shared with askeladd's PR 52, plus a per-leg "
                  "AGXAccelerator utilization gate whose first (interval) "
                  "sample is discarded and which requires 3 consecutive busy "
                  "samples. The gate is self-tested against real Metal load in "
                  "both directions before any leg is trusted.",
    "scored_surface": "Every arm patches quantized.h and its generated twin and "
                      "reverts them on every exit path. The branch keeps zero "
                      "scored-surface diff versus the frontier.",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e49-prereg.json")
    args = ap.parse_args()

    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    payload = {
        "assignment": ASSIGNMENT,
        "base_sha": BASE_SHA,
        "head_at_preregistration": head,
        "e46_refit": {"c": E46_C, "b_ms_per_stream": E46_B, "a_ms_per_row": E46_A,
                      "source": "research/qwen38-r1-e46-stream-vs-groupwidth.md, "
                                "PR #51, base 01f69e18"},
        "arm1_isolated": ARM1,
        "arm2_shared": ARM2,
        "decision_rules": RULES,
        "instrument_honesty": INSTRUMENT,
    }
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))

    print("E49 pre-registration   head=%s   base=%s" % (head[:8], BASE_SHA[:8]))
    print("arm 1: <T,9,5> vs <T,9,3>, isolated")
    print("  predicted %+.2f %% (E46 refit)   %+.2f %% (contrast B transferred)"
          % (ARM1["predicted_pct_from_e46_refit"],
             ARM1["predicted_pct_from_contrast_b"]))
    print("  local win if <= -8 %%; H_local_eaten and STOP if >= -2 %%")
    print("arm 2: ceiling dose ladder, untouched widths %s"
          % ARM2["untouched_widths"])
    for d in ARM2["doses"]:
        print("  %-12s cell_max=%-4s entry=%-4s %s"
              % (d["arm"], d["cell_max"], d["entry_heuristic"],
                 d.get("cell", "-")))
    print("  predicted +%.1f %% at 129; refuted at <= 2 %%; ladder invalid if "
          "dose_null moves > 2 %%" % ARM2["prediction_pct_at_129"])
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
