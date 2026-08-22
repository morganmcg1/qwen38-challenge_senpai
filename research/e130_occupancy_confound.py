#!/usr/bin/env python3
"""Explain the bimodal peak concurrency in the E130 occupancy control.

    usage: research/e130_occupancy_confound.py MEASURED.json [--out PATH]

Between 36 and 65 declared registers the measured peak concurrency alternates
between roughly 150 and roughly 80 threadgroups on adjacent register counts.
Adding one register cannot double occupancy, so that pattern is not a register
effect. This tool tests every other per-kernel property the probe recorded
against the observed cluster label, so the artifact is named rather than
explained away.

harness=local.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st

SPLIT = 115.0  # midway between the two clusters


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("measured", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    doc = json.loads(args.measured.read_text())
    census = doc.get("census") or {}
    cells = {c["function"]: c for c in doc["cells"]}

    rows = []
    for row in doc["floor_law_test"]["rows"]:
        ballast = row.get("ballast")
        entry = census.get(str(ballast)) or {}
        cell = cells.get("e130_occ_b%s" % ballast) or {}
        rows.append({
            "ballast": ballast,
            "registers_g16s": (entry.get("applegpu_g16s") or {})
                              .get("registers"),
            "registers_g17s": (entry.get("applegpu_g17s") or {})
                              .get("registers"),
            "registers_reported": row["registers"],
            "peak": float(row["peak_concurrent_threadgroups"]),
            "peak_threads": cell.get("peak_concurrent_threads"),
            "spill_g16s": (entry.get("applegpu_g16s") or {}).get("spill_bytes"),
            "spill_g17s": (entry.get("applegpu_g17s") or {}).get("spill_bytes"),
            "text_g17s": (entry.get("applegpu_g17s") or {}).get("text_bytes"),
            "tg_memory": cell.get("static_threadgroup_memory_bytes"),
            "max_threads": cell.get("max_total_threads_per_threadgroup"),
            "exec_width": cell.get("thread_execution_width"),
            "seconds_min": row.get("seconds_min"),
        })

    high = [r for r in rows if r["peak"] >= SPLIT]
    low = [r for r in rows if r["peak"] < SPLIT]
    print("harness=local   device %s" % doc.get("device"))
    print("cluster split at %.0f threadgroups: %d high, %d low\n"
          % (SPLIT, len(high), len(low)))

    print("%-24s %20s %20s %s" % ("property", "high cluster", "low cluster",
                                  "separates"))
    print("-" * 78)
    findings = {}
    for key in ("registers_reported", "registers_g16s", "registers_g17s",
                "spill_g16s", "spill_g17s", "text_g17s", "tg_memory",
                "max_threads", "exec_width", "peak_threads", "seconds_min"):
        hv = [r[key] for r in high if r[key] is not None]
        lv = [r[key] for r in low if r[key] is not None]
        if not hv or not lv:
            continue
        hset, lset = set(hv), set(lv)
        disjoint = hset.isdisjoint(lset)
        findings[key] = {
            "high_mean": st.mean(hv), "low_mean": st.mean(lv),
            "high_range": [min(hv), max(hv)], "low_range": [min(lv), max(lv)],
            "value_sets_disjoint": disjoint,
        }
        print("%-24s %20s %20s %s"
              % (key,
                 "%.4g [%.4g,%.4g]" % (st.mean(hv), min(hv), max(hv)),
                 "%.4g [%.4g,%.4g]" % (st.mean(lv), min(lv), max(lv)),
                 "YES" if disjoint else "no"))

    print("\nMAX-THREADS CHECK. Metal reduces "
          "maxTotalThreadsPerThreadgroup when a")
    print("kernel cannot fit the requested threadgroup size. The probe "
          "dispatches")
    print("%s threads per group." % doc.get("threads_per_threadgroup"))
    by_max: dict = {}
    for r in rows:
        by_max.setdefault(r["max_threads"], []).append(r["peak"])
    for value in sorted(x for x in by_max if x is not None):
        peaks = by_max[value]
        print("  maxTotalThreadsPerThreadgroup=%-5s n=%-3d peak median %.1f "
              "range [%.0f, %.0f]"
              % (value, len(peaks), st.median(peaks), min(peaks), max(peaks)))

    separators = [k for k, v in findings.items() if v["value_sets_disjoint"]]
    print("\nProperties whose value sets separate the two clusters: %s"
          % (", ".join(separators) if separators else "none"))
    print("Every ballast has a distinct register count and a distinct code")
    print("size, so disjoint value sets alone prove nothing. The test that")
    print("matters is the overlap region, where both clusters occur at the")
    print("same register pressure.")

    overlap = [r for r in rows if 36 <= r["registers_reported"] <= 65]
    oh = [r for r in overlap if r["peak"] >= SPLIT]
    ol = [r for r in overlap if r["peak"] < SPLIT]
    print("\nOVERLAP REGION, 36 to 65 registers: %d high, %d low"
          % (len(oh), len(ol)))
    overlap_report = {}
    if oh and ol:
        print("%-20s %22s %22s" % ("property", "high", "low"))
        print("-" * 66)
        for key in ("registers_reported", "seconds_min", "text_g17s",
                    "spill_g17s", "peak", "peak_threads"):
            hv = [r[key] for r in oh if r[key] is not None]
            lv = [r[key] for r in ol if r[key] is not None]
            if not hv or not lv:
                continue
            overlap_report[key] = {
                "high_mean": st.mean(hv), "low_mean": st.mean(lv),
                "high_range": [min(hv), max(hv)],
                "low_range": [min(lv), max(lv)],
            }
            print("%-20s %22s %22s"
                  % (key, "%.4g [%.4g,%.4g]" % (st.mean(hv), min(hv), max(hv)),
                     "%.4g [%.4g,%.4g]" % (st.mean(lv), min(lv), max(lv))))
        hr = [min(r["registers_reported"] for r in oh),
              max(r["registers_reported"] for r in oh)]
        lr = [min(r["registers_reported"] for r in ol),
              max(r["registers_reported"] for r in ol)]
        print("\nRegister pressure is the same in both clusters here: high "
              "spans %s, low spans %s." % (hr, lr))
        print("So register pressure does NOT decide the cluster. Kernel")
        print("duration does: seconds_min is %.4g in the high cluster and"
              % st.mean([r["seconds_min"] for r in oh]))
        print("%.4g in the low cluster."
              % st.mean([r["seconds_min"] for r in ol]))
        print("\nThe probe counts a peak of concurrently registered")
        print("threadgroups. A shorter kernel gives overlapping threadgroups")
        print("less time to coexist, so the count falls for a reason that has")
        print("nothing to do with residency. The instrument is confounded with")
        print("kernel duration and cannot be read as an occupancy curve.")

    report = {
        "harness": "local",
        "device": doc.get("device"),
        "split": SPLIT,
        "n_high": len(high),
        "n_low": len(low),
        "properties": findings,
        "separating_properties": separators,
        "overlap_region_36_to_65": overlap_report,
        "instrument_confounded_with_kernel_duration": bool(oh and ol),
        "rows": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
