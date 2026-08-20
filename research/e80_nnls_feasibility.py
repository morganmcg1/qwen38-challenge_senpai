#!/usr/bin/env python3
"""Feasibility check for the NNLS attribution, on the rung-2 debug leg.

Decides one thing before any further GPU time is spent: is the buffer-signature
system identifiable at the widths rung 2 will measure? Prints the fit
diagnostics and the recovered per-kernel times, and validates the fit against
the 25 buffers that DID carry a lone `affine_qmv_fast` dispatch.
"""
from __future__ import annotations

import collections
import json
import pathlib
import re
import sys

from e80_nnls import solve_kernel_times

SIG_ENTRY = re.compile(r"^(?P<name>.+)\*(?P<count>\d+)$")


def read(path):
    for line in pathlib.Path(path).read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def parse_signature(sig: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for part in sig.split(","):
        m = SIG_ENTRY.match(part)
        if not m:
            raise ValueError(f"unparsable signature element: {part!r}")
        counts[m["name"]] = counts.get(m["name"], 0) + int(m["count"])
    return counts


def collect(path, prefix):
    """Sum the delta records into {signature: [gpu_ns, buffers, dispatches]}."""
    acc = collections.defaultdict(lambda: [0, 0, 0])
    for rec in read(path):
        if rec.get("event") != "gputime":
            continue
        for key, bucket in (rec.get("signatures") or {}).items():
            if not key.startswith(prefix):
                continue
            slot = acc[key[len(prefix):]]
            slot[0] += bucket["gpu_ns"]
            slot[1] += bucket["buffers"]
            slot[2] += bucket["dispatches"]
    return acc


def main(path, prefix="w6|target_verify|"):
    acc = collect(path, prefix)
    if not acc:
        print(f"no signatures under {prefix!r} in {path}")
        return 1
    rows = [(parse_signature(sig), v[0], v[1]) for sig, v in acc.items()]
    times, diag = solve_kernel_times(rows)

    total_ns = sum(v[0] for v in acc.values())
    total_buffers = sum(v[1] for v in acc.values())
    total_dispatches = sum(v[2] for v in acc.values())
    print(f"\n{path}\n  prefix={prefix}")
    print(f"  buffers={total_buffers} dispatches={total_dispatches} "
          f"({total_dispatches/total_buffers:.2f} per buffer) "
          f"gpu={total_ns/1e6:.1f} ms")
    print(f"  fit: signatures={diag['signatures']} kernels={diag['kernels']} "
          f"rank={diag['rank']} deficient={diag['rank_deficient']} "
          f"closure={diag['closure']:.4f}")
    if diag["zero_fitted"]:
        print(f"  fitted to zero: {len(diag['zero_fitted'])} kernels")

    # Reconstruct the total from the fit: sum over kernels of t_k * dispatches.
    per_kernel_dispatches = collections.Counter()
    for sig, v in acc.items():
        for name, c in parse_signature(sig).items():
            per_kernel_dispatches[name] += c * v[1]
    rebuilt = sum(times[k] * n for k, n in per_kernel_dispatches.items())
    print(f"  reconstructed total = {rebuilt/1e6:.1f} ms "
          f"vs measured {total_ns/1e6:.1f} ms "
          f"({100*rebuilt/total_ns:.2f} %)")

    print(f"\n{'kernel':70} {'disp':>7} {'ns/disp':>9} {'ms':>9} {'share':>7}")
    ranked = sorted(per_kernel_dispatches, key=lambda k: -times[k] * per_kernel_dispatches[k])
    for name in ranked:
        n = per_kernel_dispatches[name]
        ms = times[name] * n / 1e6
        print(f"{name[:70]:70} {n:7d} {times[name]:9.0f} {ms:9.2f} "
              f"{100*ms*1e6/total_ns:6.2f}%")

    # Independent check: the lone-dispatch buffers price one kernel directly.
    lone = {sig: v for sig, v in acc.items() if sum(parse_signature(sig).values()) == 1}
    if lone:
        print(f"\nvalidation against lone-dispatch buffers "
              f"(these are NOT given extra weight in the fit):")
        print(f"{'kernel':60} {'buffers':>8} {'direct ns':>10} {'fitted ns':>10} {'ratio':>7}")
        for sig, v in sorted(lone.items(), key=lambda kv: -kv[1][1]):
            name = next(iter(parse_signature(sig)))
            direct = v[0] / v[1]
            print(f"{name[:60]:60} {v[1]:8d} {direct:10.0f} "
                  f"{times[name]:10.0f} {times[name]/direct if direct else 0:7.3f}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    args = sys.argv[1:]
    sys.exit(main(args[0], *(args[1:2])))
