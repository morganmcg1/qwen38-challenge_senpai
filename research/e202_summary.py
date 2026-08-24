#!/usr/bin/env python3
"""E202 headline aggregates: leg-level corroboration, contrasts, census."""
import json
import statistics as st
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "research/e202-session.json"
r = json.load(open(path))
legs = r["legs"]
bare = [l for l in legs if l["offset"] == "-"]
rot = [l for l in legs if l["offset"] != "-"]
b = [l["leg_mean_round_ms"] for l in bare]
q = [l["leg_mean_round_ms"] for l in rot]
c = r["contrasts_branch_serving_rounds"]

print("== LEG-LEVEL NULL / CORROBORATION ==")
print("bare SHIPPED n=%d mean=%.4f ms/round vals=%s" % (len(b), st.mean(b), [round(x, 3) for x in b]))
print("rotated      n=%d mean=%.4f ms/round sd=%.4f spread=%.4f"
      % (len(q), st.mean(q), st.stdev(q), max(q) - min(q)))
uplift = st.mean(q) - st.mean(b)
armed = st.mean([l["witness"]["armed_rounds"] for l in rot])
rounds = rot[0]["witness"]["rounds"]
serving = rot[0]["witness"]["branch_serving_rounds"]
pred = (armed / 2 * c["sync_ms_per_round"]["mean"]
        + armed / 2 * c["all_minus_shipped_ms_per_round"]["mean"]) / rounds
print("armed_rounds=%.2f of rounds=%d (serving=%d)" % (armed, rounds, serving))
print("observed leg uplift  = %.4f ms/round" % uplift)
print("predicted from triples= %.4f ms/round  -> recovery %.1f%%" % (pred, 100 * uplift / pred))

print()
print("== PARENT METRICS ==")
for name, grp in (("bare", bare), ("rotated", rot)):
    print("%-8s mtp_spt=%.6f serial_spt=%.6f speedup=%.4f"
          % (name,
             st.mean([l["mtp_seconds_per_token"] for l in grp]),
             st.mean([l["serial_seconds_per_token"] for l in grp]),
             st.mean([l["mtp_decode_speedup"] for l in grp])))
print("edl=%.4f accepted_draft_rate=%.4f decode_tokens=%d"
      % (legs[0]["effective_mean_draft_len"], legs[0]["accepted_draft_rate"], legs[0]["decode_tokens"]))

print()
print("== CONTRASTS (branch-serving rounds, n=%d non-overlapping triples, 16 calls/round) =="
      % c["sync_ms_per_round"]["n"])
R = st.mean(q)
for k in ("sync_ms_per_round", "inner_ms_per_round",
          "all_minus_shipped_ms_per_round", "net_interior_ms_per_round"):
    v = c[k]
    print("%-32s %+8.3f +/- %.3f (2sig)  %+6.2f%% of %.1f ms round"
          % (k, v["mean"], v["two_sigma"], 100 * v["mean"] / R, R))
sb, ib = c["sync_us_per_barrier"], c["inner_us_per_barrier"]
print("first barrier    = %.1f us/call" % sb)
print("marginal barrier = %.1f us/call" % ib)
print("group-drain excess = %.1f us/call -> %.3f ms/round (%.2f%% of round)"
      % (sb - ib, 16 * (sb - ib) / 1000, 100 * 16 * (sb - ib) / 1000 / R))
inner = c["inner_ms_per_round"]["mean"]
print("inner vs E198 priced 1.6 ms/round = %.2fx ; vs MUE 0.567 ms/round = %.1fx"
      % (inner / 1.6, inner / 0.567))

print()
print("== SERVED-WIDTH CENSUS (steady-state decode, warm-up excluded) ==")
cen = r["session_split_call_census_by_width"]
tot = sum(cen.values())
print("session (14 legs x 512 tokens): total %d split-cell calls" % tot)
for w in ("6", "7", "8", "9"):
    print("  qL=%s  %6d calls  %6.2f%%" % (w, cen.get(w, 0), 100 * cen.get(w, 0) / tot))
print("per-leg calls  :", legs[0]["split_call_census_by_width"])
print("per-leg rounds :", legs[0]["round_census_by_width"])

print()
print("== TEMPS ==")
ent = [float(l["gpu_temp_entry"]) for l in legs]
ext = [float(l["gpu_temp_exit"]) for l in legs]
print("entry min=%.2f max=%.2f spread=%.2f C" % (min(ent), max(ent), max(ent) - min(ent)))
print("exit  min=%.2f max=%.2f spread=%.2f C" % (min(ext), max(ext), max(ext) - min(ext)))
print("exactness: matched=%s divergence_max=%d"
      % (all(l["all_tokens_matched"] for l in legs),
         max(l["residual_divergence_count"] for l in legs)))
