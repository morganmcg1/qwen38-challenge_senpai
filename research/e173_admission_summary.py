#!/usr/bin/env python3
"""E173: print the admission census artifact as compact tables."""

from __future__ import annotations

import collections
import json
import pathlib
import sys

path = pathlib.Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else pathlib.Path(__file__).resolve().parent / "e173-artifacts/admission.json"
)
d = json.loads(path.read_text())
print({k: v for k, v in d.items() if not isinstance(v, list)})

tot = collections.defaultdict(float)
cells = collections.defaultdict(int)
acc = collections.defaultdict(int)
for r in d["routable"]:
    tot[r["m"]] += r["host_ns_per_call"] * r["calls_per_round"]
    cells[r["m"]] += r["calls_per_round"]
    acc[r["m"]] += r["calls_per_round"] if r["accepted"] else 0
print("\nroutable predicate")
for m in sorted(tot):
    print(
        f"  m={m:3d} cells={cells[m]} accepted={acc[m]} "
        f"total={tot[m] / 1e3:8.2f} us  per_cell={tot[m] / cells[m]:7.1f} ns"
    )

print("\nrefusal counterfactuals")
for r in d["refusal_counterfactuals"]:
    print(
        f"  {r['case']:22s} n={r['n']:6d} bits={r['bits']} g={r['group_size']:2d} "
        f"{r['host_ns_per_call']:8.1f} ns"
    )

print("\nxsums sidecar take")
for r in d["xsums_sidecar"]:
    print(f"  {r['state']:20s} table={r['returned_table']!s:5s} {r['host_ns_per_call']:8.1f} ns")

p = d["counter_increment_pair"]
print(f"\ncounter pair {p['host_ns_per_call']:.1f} ns")

print("\nrouted entry-point graph build (per cell, and x calls_per_round)")
per_m = collections.defaultdict(float)
for r in d["routed_entry_point_graph_build"]:
    per_m[r["m"]] += r["host_ns_per_call"] * r["calls_per_round"]
    print(
        f"  {r['shape']:14s} m={r['m']} routed={r['routed']!s:5s} "
        f"table={r['table_path']!s:5s} {r['host_ns_per_call'] / 1e3:8.2f} us/cell "
        f"x{r['calls_per_round']:3d} = {r['host_ns_per_call'] * r['calls_per_round'] / 1e6:7.3f} ms"
    )
print("\nper-round entry-point build total")
for m in sorted(per_m):
    print(f"  m={m}: {per_m[m] / 1e6:7.3f} ms")
