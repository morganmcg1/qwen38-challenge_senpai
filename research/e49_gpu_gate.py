#!/usr/bin/env python3
"""Is another process using this GPU right now?

`benchmark.sh`'s run lock is per-$HOME and every role has its own $HOME, so two
students can hold "the" lock at once (advisor, PR 53). `MLXFAST_LOCAL_RUN_LOCK_DIR`
fixes the lock; this fixes the *observation*, which the lock cannot give us: a
peer student's worker is invisible to a resident-model process scan run from
another role's process view, but its Metal work is not invisible to the
accelerator's own counters.

`AGXAccelerator -> PerformanceStatistics -> Device Utilization %` is an INTERVAL
counter accumulated since its own previous read, not a gauge. Consequences,
which this script implements rather than documents:

  * the first read after any gap covers an unbounded prior window, so it is a
    PRIMING read and is discarded;
  * one high sample is not busy -- BUSY requires `--consecutive` samples in a
    row over the threshold;
  * a missing counter is `counter_unavailable`, never `idle`.

A gate that can only return one answer is not a measurement, so `--selftest`
puts real Metal load on the device and requires the gate to say BUSY, then
requires it to say IDLE again once the load exits. It fails closed if either
direction does not reproduce.

  python3 research/e49_gpu_gate.py --samples 5
  python3 research/e49_gpu_gate.py --selftest
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time

UTIL = re.compile(r'"Device Utilization %"\s*=\s*(\d+)')
DEFAULT_PYTHON = "/opt/homebrew/bin/python3"

LOAD_SNIPPET = """
import time
import mlx.core as mx
a = mx.random.normal((2048, 2048))
b = mx.random.normal((2048, 2048))
end = time.time() + %(seconds)f
while time.time() < end:
    for _ in range(20):
        a = mx.matmul(a, b) * 1e-4
    mx.eval(a)
"""


def read_utilization() -> int | None:
    try:
        out = subprocess.run(["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    hit = UTIL.search(out)
    return int(hit.group(1)) if hit else None


def sample(samples: int, interval: float, threshold: int,
           consecutive: int) -> dict:
    priming = read_utilization()
    values: list[int] = []
    for _ in range(samples):
        time.sleep(interval)
        v = read_utilization()
        if v is None:
            return {"state": "counter_unavailable", "priming": priming,
                    "samples": values, "threshold": threshold,
                    "consecutive_required": consecutive}
        values.append(v)

    run = best = 0
    for v in values:
        run = run + 1 if v >= threshold else 0
        best = max(best, run)
    return {
        "state": "busy" if best >= consecutive else "idle",
        "priming_discarded": priming,
        "samples": values,
        "max": max(values),
        "mean": round(sum(values) / len(values), 2),
        "longest_busy_run": best,
        "threshold": threshold,
        "consecutive_required": consecutive,
        "interval_seconds": interval,
    }


def selftest(args) -> int:
    """Prove the gate can return BOTH answers before any answer is trusted."""
    load = subprocess.Popen(
        [args.python, "-c", LOAD_SNIPPET % {"seconds": args.load_seconds}],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    time.sleep(3.0)
    under_load = sample(args.samples, args.interval, args.threshold,
                        args.consecutive)
    load.wait(timeout=args.load_seconds + 60)
    time.sleep(3.0)
    after = sample(args.samples, args.interval, args.threshold, args.consecutive)

    result = {
        "load_exit_code": load.returncode,
        "load_stderr": (load.stderr.read() or "").strip().splitlines()[-3:],
        "under_load": under_load,
        "after_load": after,
        "reports_busy_under_load": under_load["state"] == "busy",
        "reports_idle_after_load": after["state"] == "idle",
    }
    result["selftest_passed"] = (result["reports_busy_under_load"]
                                 and result["reports_idle_after_load"])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["selftest_passed"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--threshold", type=int, default=25)
    ap.add_argument("--consecutive", type=int, default=3)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--load-seconds", type=float, default=20.0)
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.selftest:
        return selftest(args)

    result = sample(args.samples, args.interval, args.threshold, args.consecutive)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    print(text)
    # idle -> 0, busy -> 1, counter missing -> 2: the caller must distinguish
    # "nobody else is on the GPU" from "we could not tell".
    return {"idle": 0, "busy": 1}.get(result["state"], 2)


if __name__ == "__main__":
    sys.exit(main())
