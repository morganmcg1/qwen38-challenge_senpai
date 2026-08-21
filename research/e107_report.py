#!/usr/bin/env python3
"""Reduce one E107 isolated-dose session to per-arm statistics.

Reports, per arm, the palindrome-averaged GPU time with its standard
deviation over blocks, the implied achieved bandwidth of the 157,337,600-byte
draft-readout stream, and the difference from the shipped arm with a paired
standard error over blocks.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

FINDING36_SLOPE_US_PER_GB = 3670.2
FINDING36_FIXED_US = 9.90


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def sd(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_json")
    ap.add_argument("--json", default=None)
    ap.add_argument("--drop-first", type=int, default=0,
                    help="drop the first N blocks as warm-up")
    args = ap.parse_args()

    doc = json.loads(pathlib.Path(args.session_json).read_text())
    buffer_bytes = doc["buffer_bytes"]
    weight_bytes = doc["weight_bytes"]

    by_arm: dict[str, list[float]] = {}
    for row in doc["timing"]:
        if row["block"] < args.drop_first:
            continue
        by_arm.setdefault(row["arm"], []).append(row["gpu_us"])

    base = mean(by_arm["a_shipped"])
    law_us = FINDING36_FIXED_US + buffer_bytes / 1e9 * FINDING36_SLOPE_US_PER_GB

    print(f"session: {doc['device']} {doc['architecture']}")
    print(f"cell   : K={doc['k']} N={doc['n']} g={doc['group_size']} "
          f"threadgroups={doc['threadgroups']} tg_threads="
          f"{doc['threads_per_threadgroup']}")
    print(f"bytes  : weights {weight_bytes:,} metadata "
          f"{doc['metadata_bytes']:,} total {buffer_bytes:,}")
    print(f"Finding 36 law for this buffer: {law_us:.2f} us")
    print()
    header = (f"{'arm':<12} {'gpu_us':>10} {'sd':>8} {'blocks':>7} "
              f"{'GB/s':>8} {'vs law':>9} {'vs shipped':>11} {'paired sd':>10}")
    print(header)
    out_rows = []
    for arm, xs in by_arm.items():
        m = mean(xs)
        gbps = buffer_bytes / (m * 1e-6) / 1e9
        paired = [x - y for x, y in zip(xs, by_arm["a_shipped"])]
        row = {
            "arm": arm,
            "gpu_us_mean": m,
            "gpu_us_sd": sd(xs),
            "blocks": len(xs),
            "implied_gbps_full_buffer": gbps,
            "pct_of_law": 100.0 * m / law_us,
            "pct_vs_shipped": 100.0 * (m - base) / base,
            "paired_delta_us_mean": mean(paired),
            "paired_delta_us_sd": sd(paired),
        }
        out_rows.append(row)
        print(f"{arm:<12} {m:10.2f} {row['gpu_us_sd']:8.2f} {len(xs):7d} "
              f"{gbps:8.1f} {row['pct_of_law']:8.1f}% "
              f"{row['pct_vs_shipped']:10.2f}% {row['paired_delta_us_sd']:10.3f}")

    stream: dict[str, list[float]] = {}
    for row in doc.get("stream", []):
        if row["block"] < args.drop_first:
            continue
        stream.setdefault(row["range"], []).append(row["gbps"])
    print()
    for name, xs in stream.items():
        print(f"stream {name:<24} {mean(xs):7.1f} GB/s  sd {sd(xs):5.2f}  "
              f"n={len(xs)}")

    print()
    for row in doc.get("arm_exactness", []):
        flag = "" if row["pass"] else "   <-- FAIL"
        print(f"exactness {row['arm']:<12} expect_bit_exact="
              f"{str(row['expect_bit_exact']):<5} differing="
              f"{row['differing']}/{row['total']} worst={row['worst_abs']:.3e}"
              f"{flag}")
    fid = doc.get("fidelity")
    if fid:
        print(f"fidelity vs CPU reference: rows={fid['rows_checked']} "
              f"worst_rel={fid['worst_rel']:.3e} pass={fid['pass']}")
    ctl = doc.get("positive_control")
    if ctl:
        print(f"positive control: perturbed_rel={ctl['perturbed_rel']:.3e} "
              f"restored_rel={ctl['restored_rel']:.3e} "
              f"detected={ctl['detected']}")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(
            {"cell": {k: doc[k] for k in ("device", "architecture", "k", "n",
                                          "group_size", "threadgroups",
                                          "weight_bytes", "metadata_bytes",
                                          "buffer_bytes")},
             "law_us": law_us,
             "arms": out_rows,
             "stream_gbps": {k: mean(v) for k, v in stream.items()},
             "arm_exactness": doc.get("arm_exactness", []),
             "fidelity": fid,
             "positive_control": ctl}, indent=1))


if __name__ == "__main__":
    main()
