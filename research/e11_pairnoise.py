#!/usr/bin/env python3
"""Compare two same-arm E11 runs and say whether they differ by a uniform clock
shift or by a slow tail.

A replicate pair that exceeds the clean-pair band is only fatal if the two runs
did different work or one of them was interrupted. When the schedule is
identical and every quantile moves by the same fraction, the pair is a uniform
machine-state shift and a comparison against it is still sound, just noisier.

usage: e11_pairnoise.py RUN_A RUN_B
"""
import json
import statistics as st
import sys

LEGS = [("serial", "03-mtp-timed.json"), ("mtp", "04-mtp-timed.json")]
QUANTILES = [("p10", 0.10), ("p25", 0.25), ("p50", 0.50),
             ("p75", 0.75), ("p90", 0.90), ("p99", 0.99)]


def load(run, name):
    with open(f"{run}/reports/{name}") as fh:
        return json.load(fh)


def quantile(values, p):
    ordered = sorted(values)
    return ordered[int(p * (len(ordered) - 1))]


def pct(new, old):
    return 100.0 * (new - old) / old


def meta(run, key):
    with open(f"{run}/meta.txt") as fh:
        for line in fh:
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip()
    return "?"


def main(run_a, run_b):
    for key in ("arm", "worker_sha256", "source_sha256", "golden"):
        va, vb = meta(run_a, key), meta(run_b, key)
        flag = "" if va == vb else "   <-- DIFFERS"
        print(f"{key:16s} {va}{flag}")
    print(f"{'thermal_before':16s} A {meta(run_a, 'thermal_before')}")
    print(f"{'':16s} B {meta(run_b, 'thermal_before')}")
    print()

    for leg, name in LEGS:
        a, b = load(run_a, name), load(run_b, name)
        va = a["block_request_seconds"][1:]
        vb = b["block_request_seconds"][1:]
        print(f"--- {leg} leg   n={len(va)}/{len(vb)}")
        shifts = []
        for label, p in QUANTILES:
            xa, xb = quantile(va, p), quantile(vb, p)
            shifts.append(pct(xb, xa))
            print(f"    {label}   {xa * 1e3:8.3f} ms  {xb * 1e3:8.3f} ms  {pct(xb, xa):+7.3f} %")
        ma, mb = st.mean(va), st.mean(vb)
        print(f"    mean  {ma * 1e3:8.3f} ms  {mb * 1e3:8.3f} ms  {pct(mb, ma):+7.3f} %")
        print(f"    max   {max(va) * 1e3:8.3f} ms  {max(vb) * 1e3:8.3f} ms")
        core = shifts[:5]
        print(f"    p10-p90 shift spread {min(core):+.3f} %% .. {max(core):+.3f} %%"
              f"   (uniform shift => narrow spread)")
        print()

    a, b = load(run_a, "04-mtp-timed.json"), load(run_b, "04-mtp-timed.json")
    ea, eb = a["effective_draft_lengths"], b["effective_draft_lengths"]
    print(f"schedule identical: {ea == eb}")
    if ea != eb:
        print("  differing rounds:", sum(1 for x, y in zip(ea, eb) if x != y))
        return
    print("per-depth medians (identical schedule, so identical bins):")
    for depth in sorted({x for x in ea[1:] if x > 0}):
        xa = [t for t, e in zip(a["block_request_seconds"][1:], ea[1:]) if e == depth]
        xb = [t for t, e in zip(b["block_request_seconds"][1:], eb[1:]) if e == depth]
        print(f"    d={depth} n={len(xa):4d}  {st.median(xa) * 1e3:8.3f} ms  "
              f"{st.median(xb) * 1e3:8.3f} ms  {pct(st.median(xb), st.median(xa)):+7.3f} %")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
