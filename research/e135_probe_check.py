#!/usr/bin/env python3
"""Witness the compiled derived-cluster probe fraction off a leg's own trace.

The shipped fraction is 0.15 and the declared head has 12,292 leaves, so the
run must build ceil(0.15 * 12292) = 1844 probes. The pre-E135 fraction 0.25
would build 3073, so the two arms are separated by the recorded integer and not
by the source tree the worker happened to be built from.
"""
import json
import math
import sys

WANT_FRACTION = 0.15
WANT_COUNT = 1844
CONTROL_FRACTION = 0.25

path = sys.argv[1]
with open(path) as handle:
    trace = json.load(handle)

missing = [k for k in ("probe_fraction", "probe_leaves", "probe_count")
           if k not in trace]
if missing:
    print(f"FAIL probe witness: {path} has no {missing}; the worker predates "
          f"the probe trace field")
    sys.exit(1)

fraction = trace["probe_fraction"]
leaves = trace["probe_leaves"]
count = trace["probe_count"]
derived = max(1, math.ceil(fraction * leaves))
control = max(1, math.ceil(CONTROL_FRACTION * leaves))

print(f"probe_fraction {fraction}  probe_leaves {leaves}  probe_count {count}")
print(f"derived from the recorded pair: {derived}")
print(f"control at {CONTROL_FRACTION}: {control} (must differ)")

ok = True
if fraction != WANT_FRACTION:
    print(f"FAIL fraction {fraction} != {WANT_FRACTION}")
    ok = False
if count != WANT_COUNT:
    print(f"FAIL probe_count {count} != {WANT_COUNT}")
    ok = False
if count != derived:
    print(f"FAIL probe_count {count} != ceil(fraction * leaves) {derived}")
    ok = False
if control == count:
    print(f"FAIL positive control: {CONTROL_FRACTION} derives the same count")
    ok = False

print("PASS probe witness" if ok else "FAIL probe witness")
sys.exit(0 if ok else 1)
