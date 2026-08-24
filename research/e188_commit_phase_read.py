#!/usr/bin/env python3
"""E188 stage 2 reader. Per-round phase medians for one or more trace legs.

harness=local. Reads `research/out/<tag>/trace.txt` written by
`research/e79_trace_leg.sh` and reports the median of each round phase, with
`commit_us` first because that is the phase E185 saw grow about 2.5x.

Round 1 is dropped: it pays the cold head cache, the first kernel records and
the first shape warmups, and it is one round out of a short leg, so keeping it
would move a median that the whole bisection turns on.

Usage: python3 research/e188_commit_phase_read.py TAG [TAG ...]
"""
import json
import os
import re
import statistics as st
import sys

PHASES = ["commit_us", "readout_us", "upkeep_us", "eval_wall_us",
          "verify_build_us", "draft_build_us", "round_us"]


def read(tag):
    path = os.path.join("research/out", tag, "trace.txt")
    rounds = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith("mtp-trace:"):
                continue
            kv = dict(re.findall(r"(\w+)=([-\w.]+)", line))
            try:
                kv["round"] = int(kv["round"])
            except (KeyError, ValueError):
                continue
            rounds.append(kv)
    rounds.sort(key=lambda r: r["round"])
    return rounds[1:]


def meta(tag):
    path = os.path.join("research/out", tag, "meta.txt")
    out = {}
    if os.path.exists(path):
        for line in open(path):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                out[k] = v
    return out


def main():
    tags = sys.argv[1:]
    if not tags:
        print(__doc__)
        return
    results = {}
    for tag in tags:
        rounds = read(tag)
        m = meta(tag)
        rec = {"tag": tag, "nRounds": len(rounds),
               "rev": m.get("e188_bisect_rev", "HEAD"),
               "blob": m.get("e188_bisect_blob", "")[:10],
               "coolGate": m.get("cool_gate"),
               "gateQualified": m.get("gate_qualified_for_timing"),
               "workerSha": m.get("worker_sha256", "")[:12],
               "baseSha": m.get("base_sha", "")[:10],
               "gpuTempEntry": m.get("gpu_temp_entry_c"),
               "gpuTempExit": m.get("gpu_temp_exit_c")}
        for ph in PHASES:
            vals = [float(r[ph]) for r in rounds if ph in r]
            if vals:
                rec[ph] = st.median(vals)
                rec[ph + "_iqr"] = (st.quantiles(vals, n=4)[2]
                                    - st.quantiles(vals, n=4)[0]) if len(vals) > 3 else 0.0
        results[tag] = rec

    print(f"{'tag':22s} {'rev':10s} {'n':>4s} " +
          " ".join(f"{p.replace('_us',''):>12s}" for p in PHASES))
    for tag, rec in results.items():
        print(f"{tag:22s} {rec['rev']:10s} {rec['nRounds']:4d} " +
              " ".join(f"{rec.get(p, float('nan')):12.1f}" for p in PHASES))

    print("\ncommit_us median, IQR, and thermal record per leg:")
    for tag, rec in results.items():
        print(f"  {tag:22s} commit={rec.get('commit_us', float('nan')):7.1f} us "
              f"IQR={rec.get('commit_us_iqr', float('nan')):6.1f}  "
              f"gate_qualified={rec['gateQualified']}  "
              f"temp {rec['gpuTempEntry']}->{rec['gpuTempExit']} C  "
              f"worker={rec['workerSha']}")

    json.dump(results, open("/tmp/e188/phases.json", "w"), indent=1)
    print("\nwrote /tmp/e188/phases.json")


if __name__ == "__main__":
    main()
