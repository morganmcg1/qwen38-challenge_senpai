#!/usr/bin/env python3
"""Reduce the E87 section 8 round-level ABBA session.

The legs run in the order B K B K B K B inside one session on one binary, so the
arm is an environment gate and every other identity field is held fixed.  A
session that warms or throttles adds a drift term that is roughly linear in leg
position.  Each interior K leg sits between two B legs, so the contrast

    delta_i = K_i - (B_before + B_after) / 2

cancels any drift that is linear in position.  Averaging the three such
contrasts gives the drift-corrected effect.

The four B legs also give the session null.  Their residuals about a straight
line in leg position measure what this session can resolve, so an effect inside
that null is not distinguishable from session noise.

Usage:
    python3 research/e87_s8_abba_reduce.py <prefix> [--json OUT]
"""
import json
import os
import statistics as st
import sys

ORDER = ["base-1", "kernel-1", "base-2", "kernel-2", "base-3", "kernel-3",
         "base-4"]
FIELDS = ["mtp_seconds_per_token", "serial_seconds_per_token",
          "mtp_decode_speedup", "effective_mean_draft_len",
          "accepted_draft_rate"]


def read(prefix, leg):
    path = "research/out/%s-%s/score.json" % (prefix, leg)
    with open(path) as f:
        d = json.load(f)
    m = d["metrics"]
    row = {k: m[k] for k in FIELDS}
    row["all_tokens_matched"] = m["all_tokens_matched"]
    row["residual_divergence_count"] = m["residual_divergence_count"]
    row["head"] = m["head_provenance_sha256"]
    row["decode_tokens"] = m["decode_tokens"]
    env = "research/out/%s-%s/meta.txt" % (prefix, leg)
    row["worker"] = ""
    if os.path.exists(env):
        for line in open(env):
            if line.startswith("worker_sha256="):
                row["worker"] = line.strip().split("=", 1)[1]
    return row


def linfit_resid(ys, xs):
    n = len(ys)
    mx, my = st.fmean(xs), st.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    return a, b, resid


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "e87s8r"
    out_json = None
    if "--json" in sys.argv:
        out_json = sys.argv[sys.argv.index("--json") + 1]

    legs = [read(prefix, leg) for leg in ORDER]

    print("=== exactness and identity, every leg ===")
    ok = True
    for leg, r in zip(ORDER, legs):
        good = r["all_tokens_matched"] and r["residual_divergence_count"] == 0
        ok = ok and good
        print("%-9s matched=%-5s div=%d tokens=%d draft_len=%.15g "
              "acc=%.15g" % (leg, r["all_tokens_matched"],
                             r["residual_divergence_count"], r["decode_tokens"],
                             r["effective_mean_draft_len"],
                             r["accepted_draft_rate"]))
    workers = {r["worker"] for r in legs}
    heads = {r["head"] for r in legs}
    drafts = {r["effective_mean_draft_len"] for r in legs}
    accs = {r["accepted_draft_rate"] for r in legs}
    print("distinct worker digests %d, head digests %d" % (len(workers), len(heads)))
    print("distinct effective_mean_draft_len %d, accepted_draft_rate %d"
          % (len(drafts), len(accs)))
    print("all legs exact: %s" % ok)
    print()

    print("=== per-leg timing ===")
    print("%-9s %3s %22s %22s %14s"
          % ("leg", "pos", "mtp s/tok", "serial s/tok", "local ratio"))
    for i, (leg, r) in enumerate(zip(ORDER, legs)):
        print("%-9s %3d %22.9f %22.9f %14.6f"
              % (leg, i, r["mtp_seconds_per_token"],
                 r["serial_seconds_per_token"], r["mtp_decode_speedup"]))
    print()

    result = {"prefix": prefix, "order": ORDER, "all_legs_exact": ok,
              "distinct_worker_digests": len(workers),
              "distinct_draft_len": len(drafts)}

    for field, better in (("mtp_seconds_per_token", "lower"),
                          ("mtp_decode_speedup", "higher")):
        print("=== %s (%s is better) ===" % (field, better))
        vals = [r[field] for r in legs]
        b_idx = [0, 2, 4, 6]
        k_idx = [1, 3, 5]
        b_vals = [vals[i] for i in b_idx]
        k_vals = [vals[i] for i in k_idx]

        a, slope, resid = linfit_resid(b_vals, b_idx)
        null_sd = st.stdev(resid) * (4 / 2.0) ** 0.5  # df 2, inflate to sd
        null_sd = st.stdev(resid)
        print("base legs        %s" % "  ".join("%.9f" % v for v in b_vals))
        print("kernel legs      %s" % "  ".join("%.9f" % v for v in k_vals))
        print("base drift slope %+.9f per leg (%.4f %% per leg)"
              % (slope, 100.0 * slope / st.fmean(b_vals)))
        print("base residual sd %.9f  (%.4f %% of mean)  <- session null"
              % (null_sd, 100.0 * null_sd / st.fmean(b_vals)))

        deltas = []
        for j, ki in enumerate(k_idx):
            nb = (vals[ki - 1] + vals[ki + 1]) / 2.0
            d = vals[ki] - nb
            deltas.append(d)
            print("  K%d at pos %d vs neighbour base mean %.9f -> %+.9f "
                  "(%+.4f %%)" % (j + 1, ki, nb, d, 100.0 * d / nb))
        md = st.fmean(deltas)
        base_mean = st.fmean(b_vals)
        pct = 100.0 * md / base_mean
        sd_d = st.stdev(deltas)
        print("drift-corrected mean delta %+.9f  (%+.4f %%)" % (md, pct))
        print("spread of the three contrasts sd %.9f (%.4f %%)"
              % (sd_d, 100.0 * sd_d / base_mean))
        inside = abs(pct) < 100.0 * null_sd / base_mean
        print("effect inside the session null: %s" % inside)
        print()
        result[field] = {
            "base": b_vals, "kernel": k_vals,
            "drift_slope_per_leg": slope,
            "session_null_sd": null_sd,
            "session_null_pct": 100.0 * null_sd / base_mean,
            "contrasts": deltas,
            "delta": md, "delta_pct": pct,
            "contrast_sd": sd_d,
            "inside_session_null": inside,
        }

    if out_json:
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print("wrote %s" % out_json)


if __name__ == "__main__":
    main()
