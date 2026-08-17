#!/usr/bin/env python3
"""Research-only: per-depth round cost from the TRUSTED PARENT's own timing.

E1 fitted the head-step cost vector with forced-depth arms and a phase trace.
Neither is available on this base: the frontier deleted the forcedDepth hook,
and MLX_QWEN_MTP_TRACE cannot write from inside the worker sandbox
(`(deny file-write*)`, only /dev/null allowed) while MLXFAST_NO_SANDBOX=1 is
refused in benchmark contexts.

It turns out neither is needed. Every captured `--local-iterate` report already
carries two aligned per-round arrays produced by the trusted parent:

    block_request_seconds   [Double]  wall seconds the parent waited per round
    effective_draft_lengths [Int]     draft count the candidate actually used

The serial leg (`is_serial_control`) is a full 512-round sample of depth-0
rounds, which is exactly the C(0) denominator the h vector is expressed in, and
the MTP leg supplies C(d) for every depth its schedule visited. So the cost
curve is recoverable from arms that were run for other reasons, measured by the
parent rather than by editable instrumentation.

Read the caveat with the numbers: natural-schedule rounds at depth d are a
SELECTED sample (the scheduler picked d because acceptance looked good there)
and they include whatever rollback or GDN replay a rejection triggered, whereas
E1 priced full-accept rounds only. These marginals are therefore realised
average costs per round at depth d, not E1's clean full-accept marginals.

usage: research/e11_marginal.py RUN_DIR [RUN_DIR ...]
"""

import json
import statistics
import sys
from pathlib import Path

E1_VECTOR = [0.0842, 0.0775, 0.2426, 0.3754, 0.2919, 0.3000, 0.2870, 0.3909]


def load(run: Path):
    """Return (serial_rounds, mtp_rounds) as lists of (depth, seconds)."""
    legs = {}
    for path in sorted(run.glob("reports/*-mtp-timed.json")):
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        secs = doc.get("block_request_seconds")
        depths = doc.get("effective_draft_lengths")
        if not isinstance(secs, list) or not isinstance(depths, list):
            continue
        if len(secs) != len(depths):
            raise SystemExit(f"{path}: {len(secs)} times vs {len(depths)} depths")
        key = "serial" if doc.get("is_serial_control") else "mtp"
        # Round 0 carries seed prefill and first-touch warmup on both legs.
        legs.setdefault(key, []).extend(list(zip(depths, secs))[1:])
    return legs.get("serial", []), legs.get("mtp", [])


def summarise(rounds):
    by_depth = {}
    for depth, sec in rounds:
        by_depth.setdefault(depth, []).append(sec)
    return {d: sorted(v) for d, v in sorted(by_depth.items())}


def stats(vals):
    n = len(vals)
    mean = statistics.fmean(vals)
    sd = statistics.stdev(vals) if n > 1 else float("nan")
    return n, mean, statistics.median(vals), sd / (n**0.5) if n > 1 else float("nan")


def report(run: Path):
    serial, mtp = load(run)
    if not serial or not mtp:
        print(f"{run.name}: missing serial or mtp leg; skipped")
        return
    print(f"\n=== {run.name} ===")
    meta = run / "meta.txt"
    if meta.exists():
        keep = ("arm", "source_sha256", "head_sha", "golden", "thermal_before")
        for line in meta.read_text().splitlines():
            if line.split("=", 1)[0] in keep:
                print(f"  {line}")

    # C(0): the serial leg is a pure depth-0 sample. The median is the headline
    # because per-round wall time has a long right tail (scheduler hiccups,
    # foreign load) that a mean would let leak into every normalised ratio.
    s_depths = summarise(serial)
    if list(s_depths) != [0]:
        print(f"  warning: serial leg is not pure depth 0: {list(s_depths)}")
    s0 = [sec for _, sec in serial]
    n0, mean0, med0, se0 = stats(s0)
    print(f"  C(0) serial      n={n0:4d} median={med0*1000:8.3f}ms "
          f"mean={mean0*1000:8.3f}ms se={se0*1000:6.3f}ms")

    curve = {}
    print("  depth      n   median(ms)     mean(ms)   se(ms)   C(d)/C(0)")
    for depth, vals in summarise(mtp).items():
        n, mean, med, se = stats(vals)
        curve[depth] = med
        print(f"  {depth:5d} {n:6d} {med*1000:12.3f} {mean*1000:12.3f} "
              f"{se*1000:8.3f} {med/med0:11.4f}")

    # h[d] is the marginal of the (d+1)-th head step, so it needs C(d) and
    # C(d+1) from the same run. A depth the schedule never visited leaves a
    # gap rather than an interpolation.
    print("  marginal h[d] = (C(d+1) - C(d)) / C(0)   [E1 fit in brackets]")
    curve[0] = med0
    for depth in sorted(curve):
        nxt = curve.get(depth + 1)
        if nxt is None:
            continue
        h = (nxt - curve[depth]) / med0
        old = E1_VECTOR[depth] if depth < len(E1_VECTOR) else float("nan")
        print(f"    h[{depth}] = {h:+.4f}   [{old:.4f}]   ratio={h/old:6.2f}x")


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    for arg in argv:
        report(Path(arg))


if __name__ == "__main__":
    main(sys.argv[1:])
