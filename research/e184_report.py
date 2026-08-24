#!/usr/bin/env python3
"""Print the E184 model report as the tables that go into the result."""

import json
import sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1
                   else "research/e184-model-report.json"))

print("forwards %d   gemm %.6f s   non_gemm %.6f s"
      % (d["forwards"], d["gemm_seconds"], d["non_gemm_seconds"]))
print()
print("%-26s %10s %8s %7s %8s %9s %7s"
      % ("phase", "mean_s", "share%", "sd%", "calls", "TFLOP", "%peak"))
for r in d["phase_table"]:
    peak = "%.1f" % r["pct_of_peak"] if "pct_of_peak" in r else "-"
    print("%-26s %10.6f %8.3f %7.2f %8.0f %9.3f %7s"
          % (r["phase"], r["mean_seconds"], r["share_of_bracketed_pct"],
             r["rel_sd_pct"], r["calls_per_forward"], r.get("tflop", 0), peak))

a = d["e16_residual_attribution"]
print()
print("E16 ATTRIBUTION")
print("  GEMM    measured %7.3f %%  vs E16 ceiling+residual %7.3f %%  -> %+.3f pp"
      % (a["measured_gemm_share_pct"],
         a["e16_gemm_at_ceiling_plus_residual_pct"],
         a["gemm_budget_disagreement_pp"]))
print("  nonGEMM measured %7.3f %%  vs E16 non-GEMM        %7.3f %%  -> %+.3f pp"
      % (a["measured_non_gemm_share_pct"], a["e16_non_gemm_pct"],
         a["non_gemm_budget_disagreement_pp"]))
print("  aggregate %.3f TFLOP/s = %.2f %% of peak %.3f TFLOP/s"
      % (a["aggregate_achieved_tflops"], a["aggregate_pct_of_peak"],
         a["peak_tflops_from_e65"]))
print("  implied E16 ceiling rate %.3f TFLOP/s" % a["implied_e16_ceiling_tflops"])

r = d["reconciliation"]
print()
print("RECONCILIATION")
for k in ("bracketed_total_mean_seconds",
          "trusted_seed_prefill_seconds_instrumented",
          "unbracketed_remainder_seconds", "unbracketed_remainder_pct",
          "trusted_seed_prefill_seconds_off_mode", "instrument_overhead_pct",
          "serialized_excess_over_off_mode_seconds"):
    print("  %-44s %.6f" % (k, r[k]))

L = d["mue_ladder"]
print()
print("MUE %.2f %% published = %.3f %% prefill = %.4f s"
      % (L["mue_published_pct"], L["prefill_reduction_needed_for_mue_pct"],
         L["prefill_seconds_needed_for_mue"]))
print("%-58s %9s %8s %10s %6s"
      % ("mechanism", "saved_s", "prefill%", "published%", "MUE"))
for x in L["ladder"]:
    print("%-58s %9.4f %8.3f %10.4f %6s"
          % (x["mechanism"][:58], x["prefill_seconds_saved"],
             x["prefill_reduction_pct"], x["published_score_gain_pct"],
             "PASS" if x["clears_mue"] else "fail"))
