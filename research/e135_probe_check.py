#!/usr/bin/env python3
"""Witness the compiled derived-cluster probe rung off a leg's own trace.

The declared head has 12,292 leaves, so each rung derives its own integer:
0.25 -> 3073, 0.15 -> 1844, 0.10 -> 1230. The gate asserts the integer, not the
source tree the worker happened to be built from, and it names a control rung
whose integer must differ so a check that cannot fail cannot pass.

Usage:
    e135_probe_check.py PIPELINES.JSON [--arm p10]
"""
import argparse
import json
import math
import sys

ARMS = {"p25": 0.25, "p15": 0.15, "p10": 0.10}
COUNTS = {"p25": 3073, "p15": 1844, "p10": 1230}
CONTROL = {"p10": "p15", "p15": "p25", "p25": "p15"}

parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("--arm", default="p10", choices=sorted(ARMS))
args = parser.parse_args()

want_arm = args.arm
want_fraction = ARMS[want_arm]
want_count = COUNTS[want_arm]
control_arm = CONTROL[want_arm]

with open(args.path) as handle:
    trace = json.load(handle)

missing = [k for k in ("probe_fraction", "probe_leaves", "probe_count",
                       "probe_arm", "default_probe") if k not in trace]
if missing:
    print(f"FAIL probe witness: {args.path} has no {missing}; the worker "
          f"predates the probe-arm trace fields")
    sys.exit(1)

fraction = trace["probe_fraction"]
leaves = trace["probe_leaves"]
count = trace["probe_count"]
arm = trace["probe_arm"]
default_probe = trace["default_probe"]
derived = max(1, math.ceil(fraction * leaves))
control = max(1, math.ceil(ARMS[control_arm] * leaves))

print(f"probe_arm {arm}  probe_fraction {fraction}  probe_leaves {leaves}  "
      f"probe_count {count}")
print(f"default_probe {default_probe}")
print(f"derived from the recorded pair: {derived}")
print(f"control rung {control_arm}: {control} (must differ)")

ok = True
if arm != want_arm:
    print(f"FAIL probe_arm {arm!r} != {want_arm!r}")
    ok = False
if fraction != want_fraction:
    print(f"FAIL fraction {fraction} != {want_fraction}")
    ok = False
if count != want_count:
    print(f"FAIL probe_count {count} != {want_count}")
    ok = False
if count != derived:
    print(f"FAIL probe_count {count} != ceil(fraction * leaves) {derived}")
    ok = False
if control == count:
    print(f"FAIL positive control: rung {control_arm} derives the same count")
    ok = False

# A leg that overrides the rung still reports the compiled default, so the
# witness literal must always name the rung the ranked runner would take.
if default_probe != "e135_default_probe/p10":
    print(f"FAIL default_probe {default_probe!r} does not name the p10 rung; "
          f"a run that exports nothing would not take 0.10")
    ok = False

print("PASS probe witness" if ok else "FAIL probe witness")
sys.exit(0 if ok else 1)
