#!/usr/bin/env python3
"""Sample Apple GPU utilization to decide whether another job is on this GPU.

Cross-student GPU coordination on this host cannot rely on the benchmark run
lock: benchmark.sh derives the lock path from ${HOME}, and each role has its own
${HOME}, so the lock serialises a role against itself and gives no mutual
exclusion against a neighbour. Checking for a resident model process is also
insufficient, because a QMV microbenchmark holds only tens of megabytes and
never looks like a model.

AGXAccelerator's PerformanceStatistics exposes "Device Utilization %", which is
driver-reported GPU busy time and therefore sees any process's GPU work,
including a neighbour's microbenchmark. A timed session is continuous GPU work,
so a low maximum across a short window is positive evidence that the GPU is free
rather than mere absence of evidence.

    research/gpu_busy_check.py [--seconds 12] [--threshold 5]

Exit 0 if the GPU looks idle, 1 if it looks busy, so a runner can gate on it.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time

FIELD = re.compile(r'"Device Utilization %"=(\d+)')


def sample() -> int | None:
    out = subprocess.run(
        ["ioreg", "-r", "-d", "1", "-c", "AGXAccelerator"],
        capture_output=True, text=True, check=False).stdout
    hits = [int(m) for m in FIELD.findall(out)]
    return max(hits) if hits else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--threshold", type=int, default=5,
                    help="max busy %% still considered idle")
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    samples: list[int] = []
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        value = sample()
        if value is None:
            print("gpu_busy_check: no AGXAccelerator utilization counter")
            return 2
        samples.append(value)
        time.sleep(args.interval)

    peak = max(samples)
    mean = sum(samples) / len(samples)
    busy = peak > args.threshold
    print(f"gpu_busy_check: n={len(samples)} over {args.seconds:.0f}s  "
          f"peak={peak}%  mean={mean:.1f}%  threshold={args.threshold}%  "
          f"verdict={'BUSY' if busy else 'IDLE'}")
    print(f"gpu_busy_check: samples={samples}")
    return 1 if busy else 0


if __name__ == "__main__":
    sys.exit(main())
