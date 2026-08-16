#!/usr/bin/env python3
"""Recompute the Part A per-width row-gate aggregate across the 512-token arms.

Reads research/analysis-run{I,J,K,L,M,N,O}.json and prints the per-width
compared / value-mismatch / top1-id-mismatch totals plus the per-run width-4 and
width-9 breakdowns that the result document quotes.
"""

import collections
import json
import os

RUNS = ["I", "J", "K", "L", "M", "N", "O"]
HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    totals = collections.defaultdict(lambda: [0, 0, 0])
    per_run = {}

    for run in RUNS:
        with open(os.path.join(HERE, f"analysis-run{run}.json")) as handle:
            doc = json.load(handle)
        gate = doc["row_gate"]
        per_run[run] = {}
        for width, entry in gate["per_width"].items():
            compared = entry["compared"]
            value_mm = entry.get("value_mismatches", 0)
            id_mm = entry.get("top1_id_mismatches", entry.get("id_mismatches", 0))
            bucket = totals[int(width)]
            bucket[0] += compared
            bucket[1] += value_mm
            bucket[2] += id_mm
            per_run[run][int(width)] = (compared, value_mm, id_mm, entry.get("bit_exact"))
        print(
            f"{run} {doc['label']} compared_rows={gate['compared_rows']} "
            f"unmatched_positions={gate.get('unmatched_positions')}"
        )

    print("\n| width | rows compared | value mismatches | top-1 id mismatches |")
    print("|---|---|---|---|")
    grand = [0, 0, 0]
    for width in sorted(totals):
        compared, value_mm, id_mm = totals[width]
        grand[0] += compared
        grand[1] += value_mm
        grand[2] += id_mm
        print(f"| {width} | {compared} | {value_mm} | {id_mm} |")
    print(f"| total | {grand[0]} | {grand[1]} | {grand[2]} |")

    for width in (4, 9):
        print(f"\nper-run width {width} (compared, value_mm, id_mm, bit_exact):")
        for run in RUNS:
            print(f"  {run}: {per_run[run].get(width)}")


if __name__ == "__main__":
    main()
